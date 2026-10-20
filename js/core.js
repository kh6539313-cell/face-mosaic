// Shared pieces for the image and video tools: face detection, effects, and
// the drag/resize editor for mosaic regions.

// Bundled locally (@mediapipe/tasks-vision 0.10.14) so the app works offline.
const LIB_URL   = new URL("../vendor/vision_bundle.mjs", import.meta.url).href;
const WASM_URL  = new URL("../vendor/wasm", import.meta.url).href;
const MODEL_URL = new URL("../vendor/blaze_face_short_range.tflite", import.meta.url).href;

export const $ = (id) => document.getElementById(id);

/* ---------- Detector ---------- */
let detectorPromise = null;
export let detectorReady = false;
export function getDetector() {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      // Imported lazily so the UI works even while the library is still downloading.
      const { FilesetResolver, FaceDetector } = await import(LIB_URL);
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      // CPU only: the GPU delegate silently returns no faces on some Android
      // GPUs, and for this tiny model the CPU is just as fast.
      return await FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.5,
      });
    })();
    detectorPromise.then(() => { detectorReady = true; }, () => { detectorPromise = null; });
  }
  return detectorPromise;
}

// 感度 1..9 → 信頼度しきい値 0.85..0.25
export const sensitivityToConfidence = (s) => 0.85 - (Number(s) - 1) * 0.075;
// Tiles produce more false positives (patterns, hands, etc.), so they need a higher score.
export const tileConfidence = (conf) => Math.min(0.95, conf + 0.2);

const tile = document.createElement("canvas");
const tctx = tile.getContext("2d", { willReadFrequently: false });

/* Detect on the full image and on overlapping tiles at several scales, so that
   small faces are also found. Returns raw boxes (not merged) with a `tile` flag.
   `maxLevels` limits how many tile scales are tried (video uses fewer for speed). */
let lastMinConf = null;
async function detectorWith(minConf) {
  const det = await getDetector();
  if (minConf !== lastMinConf) {
    await det.setOptions({ minDetectionConfidence: minConf });
    lastMinConf = minConf;
  }
  return det;
}

// Runs the model on one rectangle of `src` and appends hits (in src pixels) to `boxes`.
function detectRect(det, src, sx, sy, sw, sh, isTile, boxes) {
  // Downscale big tiles; the model works on 128px input anyway.
  const scale = Math.min(1, 640 / Math.max(sw, sh));
  tile.width = Math.max(1, Math.round(sw * scale));
  tile.height = Math.max(1, Math.round(sh * scale));
  tctx.drawImage(src, sx, sy, sw, sh, 0, 0, tile.width, tile.height);
  for (const d of det.detect(tile).detections) {
    const b = d.boundingBox;
    // faces that fill most of a tile are left to coarser passes
    if (isTile && b.width / tile.width > 0.6) continue;
    boxes.push({
      x: sx + b.originX / scale,
      y: sy + b.originY / scale,
      w: b.width / scale,
      h: b.height / scale,
      score: d.categories?.[0]?.score ?? 0,
      tile: isTile,
    });
  }
}

/* Detects only inside the given rectangles (used to follow known faces in video). */
export async function detectInRects(src, rects, { minConf }) {
  const det = await detectorWith(minConf);
  const boxes = [];
  for (const r of rects) detectRect(det, src, r.x, r.y, r.w, r.h, true, boxes);
  return boxes;
}

export async function detectRaw(src, { minConf, maxLevels = Infinity, yieldEvery = true } = {}) {
  const det = await detectorWith(minConf);
  const W = src.width, H = src.height;
  const boxes = [];
  const run = (sx, sy, sw, sh, isTile) => detectRect(det, src, sx, sy, sw, sh, isTile, boxes);

  run(0, 0, W, H, false);
  // The model needs a face to span roughly 15%+ of its input, so shrink the
  // tile size step by step (down to ~1/12 of the image) to catch small faces.
  const minSide = Math.min(W, H);
  const minTile = Math.max(200, Math.max(W, H) / 12);
  let lastTile = 0, level = 0;
  for (let size = Math.max(W, H) * 0.6; size >= minTile && level < maxLevels; size *= 0.6) {
    const s = Math.round(Math.min(size, minSide));
    if (s === lastTile) continue;
    lastTile = s;
    level++;
    const step = Math.round(s * 0.7);
    for (let y = 0; ; y += step) {
      const yy = Math.min(y, H - s);
      for (let x = 0; ; x += step) {
        const xx = Math.min(x, W - s);
        run(xx, yy, s, s, true);
        if (xx >= W - s) break;
      }
      if (yy >= H - s) break;
      if (yieldEvery) await new Promise(r => setTimeout(r)); // keep UI responsive
    }
  }
  return boxes;
}

/* Keep boxes that pass the threshold for their kind, then merge overlaps. */
export function filterAndMerge(raw, conf) {
  const tmin = tileConfidence(conf);
  return mergeBoxes(raw.filter(b => b.score >= (b.tile ? tmin : conf)));
}

/* Merge overlapping boxes into their union (covering more is safer for privacy). */
export function mergeBoxes(boxes) {
  const out = boxes.map(b => ({ ...b }));
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < out.length && !changed; i++) {
      for (let j = i + 1; j < out.length; j++) {
        const a = out[i], b = out[j];
        if (overlapRatio(a, b) > 0.35) {
          const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
          out[i] = {
            x, y,
            w: Math.max(a.x + a.w, b.x + b.w) - x,
            h: Math.max(a.y + a.h, b.y + b.h) - y,
            score: Math.max(a.score, b.score),
          };
          out.splice(j, 1);
          changed = true;
          break;
        }
      }
    }
  }
  return out;
}

/* Intersection over the smaller box's area. */
export function overlapRatio(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return ix * iy / Math.max(1e-9, Math.min(a.w * a.h, b.w * b.h));
}

/* Detected box → mosaic region (expanded, more toward forehead/hair), clamped to W×H. */
export function padBox(f, padPercent, W, H) {
  const p = padPercent / 100;
  let x = f.x - f.w * p * 0.5;
  let y = f.y - f.h * p * 0.9;
  let w = f.w * (1 + p);
  let h = f.h * (1 + p * 1.3);
  x = Math.max(0, x); y = Math.max(0, y);
  w = Math.min(W - x, w); h = Math.min(H - y, h);
  return { x, y, w, h };
}

/* ---------- Effects ---------- */
const scratch = document.createElement("canvas");
const sctx = scratch.getContext("2d");

/* opts: { effect: "mosaic"|"blur"|"fill", shape: "rect"|"ellipse", strength: 1..10 } */
export function applyEffect(g, src, r, opts) {
  const x = Math.round(r.x), y = Math.round(r.y);
  const w = Math.max(1, Math.round(r.w)), h = Math.max(1, Math.round(r.h));
  // strength 1..10 → number of blocks across the shorter side: 24..4
  const blocks = Math.round(24 - (opts.strength - 1) * (20 / 9));
  const block = Math.max(2, Math.min(w, h) / blocks);

  g.save();
  if (opts.shape === "ellipse") {
    g.beginPath();
    g.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
    g.clip();
  }

  if (opts.effect === "fill") {
    g.fillStyle = "#000";
    g.fillRect(x, y, w, h);
  } else if (opts.effect === "mosaic") {
    const sw = Math.max(1, Math.round(w / block)), sh = Math.max(1, Math.round(h / block));
    scratch.width = sw; scratch.height = sh;
    sctx.imageSmoothingEnabled = true;
    sctx.drawImage(src, x, y, w, h, 0, 0, sw, sh);
    g.imageSmoothingEnabled = false;
    g.drawImage(scratch, 0, 0, sw, sh, x, y, w, h);
  } else {
    // Blur: downscale with smoothing, then upscale with smoothing (two passes for softness).
    const f = block * 0.8;
    const sw = Math.max(1, Math.round(w / f)), sh = Math.max(1, Math.round(h / f));
    scratch.width = sw; scratch.height = sh;
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(src, x, y, w, h, 0, 0, sw, sh);
    sctx.drawImage(scratch, 0, 0, sw, sh, 0, 0, Math.max(1, sw >> 1), Math.max(1, sh >> 1));
    sctx.drawImage(scratch, 0, 0, Math.max(1, sw >> 1), Math.max(1, sh >> 1), 0, 0, sw, sh);
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    g.drawImage(scratch, 0, 0, sw, sh, x, y, w, h);
  }
  g.restore();
}

/* ---------- Frames overlay ---------- */
/* Draws region outlines with ON/OFF labels, handles on the selected region and
   the dashed preview of a region being drawn. Region: { x, y, w, h, on, label? }. */
export function drawFrames(g, canvas, regions, selected, newRect) {
  // keep lines ~2 CSS px regardless of how much the canvas is scaled down
  const k = canvas.width / Math.max(1, canvas.getBoundingClientRect().width);
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#ff4d6d";
  for (const r of regions) {
    const sel = r === selected;
    g.save();
    g.lineWidth = (sel ? 3.5 : 2) * k;
    g.setLineDash(r.on ? [] : [6 * k, 5 * k]);
    g.strokeStyle = r.on ? accent : "#bbbbc0";
    g.shadowColor = "rgba(0,0,0,.5)";
    g.shadowBlur = 3 * k;
    g.strokeRect(r.x, r.y, r.w, r.h);
    const label = (r.label ? r.label + " " : "") + (r.on ? "ON" : "OFF");
    g.font = `600 ${11 * k}px system-ui, sans-serif`;
    const tw = g.measureText(label).width + 8 * k;
    const th = 16 * k;
    const ly = r.y - th >= 0 ? r.y - th : r.y;
    g.shadowBlur = 0;
    g.setLineDash([]);
    g.fillStyle = r.on ? accent : "#6b6b70";
    g.fillRect(r.x, ly, tw, th);
    g.fillStyle = "#fff";
    g.textBaseline = "middle";
    g.fillText(label, r.x + 4 * k, ly + th / 2);
    g.restore();
  }
  if (selected && regions.includes(selected) && selected.editable !== false) {
    const s = selected, hs = 9 * k;
    g.save();
    g.fillStyle = "#fff";
    g.strokeStyle = accent;
    g.lineWidth = 2 * k;
    for (const [cx, cy] of [[s.x, s.y], [s.x + s.w, s.y], [s.x, s.y + s.h], [s.x + s.w, s.y + s.h]]) {
      g.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
      g.strokeRect(cx - hs / 2, cy - hs / 2, hs, hs);
    }
    g.restore();
  }
  if (newRect) {
    g.save();
    g.lineWidth = 2 * k;
    g.setLineDash([6 * k, 4 * k]);
    g.strokeStyle = accent;
    g.fillStyle = "rgba(255,77,109,.15)";
    g.fillRect(newRect.x, newRect.y, newRect.w, newRect.h);
    g.strokeRect(newRect.x, newRect.y, newRect.w, newRect.h);
    g.restore();
  }
}

/* ---------- Region editor (pointer interaction) ----------
   Click a region: onToggle. Drag inside an editable region: move. Drag a corner:
   resize. Drag on empty space: onAdd(rect). Regions are in canvas pixel space and
   must keep their object identity between calls of getRegions() to stay selected.
   Regions with `editable: false` can be clicked but not moved/resized/deleted. */
export function createRegionEditor(canvas, { getRegions, onToggle, onAdd, onEdit, onDelete, onChange }) {
  const ed = { selected: null, drag: null };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const toCanvas = (e) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };
  // Hit-test priority: the selected region first, then topmost first.
  const order = () => {
    const rs = getRegions();
    const out = rs.slice().reverse();
    if (ed.selected && rs.includes(ed.selected)) {
      out.splice(out.indexOf(ed.selected), 1);
      out.unshift(ed.selected);
    }
    return out;
  };
  const hitRegion = (p) => order().find(r => p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) || null;
  // Corner of any editable region within grab distance (bigger for fingers).
  const hitCorner = (p, pointerType) => {
    const k = canvas.width / Math.max(1, canvas.getBoundingClientRect().width);
    const tol = (pointerType === "touch" ? 22 : 12) * k;
    for (const r of order()) {
      if (r.editable === false) continue;
      const corners = { nw: [r.x, r.y], ne: [r.x + r.w, r.y], sw: [r.x, r.y + r.h], se: [r.x + r.w, r.y + r.h] };
      for (const [name, [cx, cy]] of Object.entries(corners)) {
        if (Math.abs(p.x - cx) <= tol && Math.abs(p.y - cy) <= tol) return { r, corner: name };
      }
    }
    return null;
  };

  // The rectangle currently being drawn, for drawFrames().
  ed.newRect = () => {
    const d = ed.drag;
    if (!d || !d.moved || d.mode !== "new") return null;
    return { x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1), w: Math.abs(d.x1 - d.x0), h: Math.abs(d.y1 - d.y0) };
  };

  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !canvas.width) return;
    canvas.setPointerCapture(e.pointerId);
    const p = toCanvas(e);
    const d = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, moved: false, cx: e.clientX, cy: e.clientY, mode: "new", r: null };
    const c = hitCorner(p, e.pointerType);
    if (c) {
      Object.assign(d, { mode: "resize", r: c.r, corner: c.corner });
    } else {
      const r = hitRegion(p);
      if (r) Object.assign(d, { mode: r.editable === false ? "click" : "move", r });
    }
    if (d.r) d.orig = { x: d.r.x, y: d.r.y, w: d.r.w, h: d.r.h };
    ed.drag = d;
  });

  canvas.addEventListener("pointermove", (e) => {
    const d = ed.drag;
    if (!d) {
      if (!canvas.width) return;
      const p = toCanvas(e);
      const c = hitCorner(p, e.pointerType);
      const r = c ? null : hitRegion(p);
      canvas.style.cursor = c ? (c.corner === "nw" || c.corner === "se" ? "nwse-resize" : "nesw-resize")
        : r ? (r.editable === false ? "pointer" : "move") : "crosshair";
      return;
    }
    const p = toCanvas(e);
    d.x1 = clamp(p.x, 0, canvas.width);
    d.y1 = clamp(p.y, 0, canvas.height);
    if (Math.hypot(e.clientX - d.cx, e.clientY - d.cy) > 5) d.moved = true;
    if (!d.moved) return;

    if (d.mode === "move" || d.mode === "resize") {
      const r = d.r, o = d.orig;
      if (d.mode === "move") {
        r.x = clamp(o.x + d.x1 - d.x0, 0, canvas.width - o.w);
        r.y = clamp(o.y + d.y1 - d.y0, 0, canvas.height - o.h);
      } else {
        // the opposite corner stays fixed
        const fx = d.corner.includes("w") ? o.x + o.w : o.x;
        const fy = d.corner.includes("n") ? o.y + o.h : o.y;
        r.x = Math.min(fx, d.x1); r.y = Math.min(fy, d.y1);
        r.w = Math.max(8, Math.abs(d.x1 - fx)); r.h = Math.max(8, Math.abs(d.y1 - fy));
      }
      ed.selected = r;
      onEdit?.(r);
    }
    onChange();
  });

  const end = () => {
    const d = ed.drag;
    ed.drag = null;
    if (!d) return;
    if (d.moved) {
      if (d.mode === "new") {
        const rect = { x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1), w: Math.abs(d.x1 - d.x0), h: Math.abs(d.y1 - d.y0) };
        if (rect.w > 4 && rect.h > 4) ed.selected = onAdd(rect) || null;
      }
    } else if (d.r) {
      onToggle(d.r);
      ed.selected = d.r;
    } else {
      ed.selected = null;
    }
    onChange();
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", () => { ed.drag = null; onChange(); });

  ed.deleteSelected = () => {
    const r = ed.selected;
    if (!r || r.editable === false) return false;
    onDelete(r);
    ed.selected = null;
    onChange();
    return true;
  };
  ed.canDelete = () => !!ed.selected && ed.selected.editable !== false && getRegions().includes(ed.selected);
  return ed;
}

/* ---------- Misc ---------- */
export function downloadBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

export const escapeHtml = (s) => String(s).replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c]));
