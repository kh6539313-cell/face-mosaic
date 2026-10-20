// Video tool (up to ~1 minute): analyze every frame for faces, link detections
// into per-person tracks (filling short misses so a face never flashes through),
// preview with per-person ON/OFF and fixed regions, then export MP4 with the
// original audio copied over.
import {
  $, detectorReady, detectRaw, detectInRects, filterAndMerge, overlapRatio, sensitivityToConfidence,
  padBox, applyEffect, drawFrames, createRegionEditor, downloadBlob, escapeHtml,
} from "./core.js";

const MB_URL = new URL("../vendor/mediabunny.min.mjs", import.meta.url).href;
let MB = null;
const loadMB = async () => (MB ??= await import(MB_URL));

const MAX_DURATION = 65;      // seconds (1分 + 少し余裕)
const ANALYSIS_FPS = 30;      // analyze at most this many frames per second
const ANALYSIS_MAX_SIDE = 960;
const OUTPUT_MAX_SIDE = 1920; // 4K is written as Full HD to keep phones from running out of memory
const STORE_CONF = 0.25;      // keep all candidates; the sensitivity slider filters later
const FOLLOW_CONF = 0.55;     // detections confident enough to keep looking around them in later frames
const DEEP_EVERY = 8;         // run the fine-tile scan every N analyzed frames
const FORGET_FRAMES = 15;     // stop following a face after this many frames without seeing it
const GAP_SEC = 1.0;          // a person missed for up to this long keeps the same track (gap is interpolated)
const HOLD_SEC = 0.4;         // keep the mosaic this long before the first / after the last detection
const MOTION_MARGIN = 10;     // extra padding (%) so fast movement doesn't peek out

const view = $("vview");
const ctx = view.getContext("2d");
const video = $("vvideo");

const v = {
  file: null,
  W: 0, H: 0,          // output / preview size (even numbers)
  duration: 0,
  frames: [],          // { t, raw: [{ x, y, w, h, score, tile }] } — normalized 0..1, t relative to the first frame
  tracks: [],          // { id, label, on, obs: [{ i, box, score }], thumb }
  perFrame: [],        // perFrame[i] = [{ track, x, y, w, h }] normalized, unpadded
  fixed: [],           // hand-drawn regions applied to every frame (canvas pixels)
  busy: false,
  cancel: null,
  thumbSeq: 0,
};

const effectOpts = () => ({
  effect: document.querySelector("input[name=veffect]:checked").value,
  shape: document.querySelector("input[name=vshape]:checked").value,
  strength: Number($("vstrength").value),
});

function setStatus(html, busy = false) {
  $("vstatusText").innerHTML = html;
  $("vstatus").classList.toggle("busy", busy);
}
function setProgress(frac, label) {
  $("vprogress").hidden = frac == null;
  if (frac != null) {
    $("vprogressBar").style.width = `${Math.round(frac * 100)}%`;
    $("vprogressText").textContent = label || "";
  }
}
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

let wakeLock = null;
async function keepAwake(on) {
  try {
    if (on && "wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen");
    if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
  } catch { /* not critical */ }
}

/* ---------- Loading ---------- */
export async function loadVideoFile(file) {
  if (v.busy) return;
  if (!file || !(file.type.startsWith("video/") || /\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name))) {
    showError("動画ファイルを選んでください。");
    return;
  }
  if (typeof VideoDecoder === "undefined" || typeof VideoEncoder === "undefined") {
    showError("このブラウザは動画の処理に対応していません。Chrome / Edge / Safari の最新版でお試しください。");
    return;
  }
  setStatus("動画を読み込み中…", true);
  $("vdrop").style.display = "none";
  $("vwork").classList.add("show");
  resetVideo();
  try {
    await loadMB();
    const input = new MB.Input({ source: new MB.BlobSource(file), formats: MB.ALL_FORMATS });
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new UserError("動画の映像が見つかりませんでした。");
    const duration = await input.computeDuration();
    if (duration > MAX_DURATION) {
      throw new UserError(`1分までの動画に対応しています（この動画は ${fmtTime(duration)}）。短く切ってからお試しください。`);
    }
    if (!(await track.canDecode())) throw new UserError("この動画の形式は、この端末では読み込めません。");

    const scale = Math.min(1, OUTPUT_MAX_SIDE / Math.max(track.displayWidth, track.displayHeight));
    Object.assign(v, {
      file, duration,
      W: Math.round(track.displayWidth * scale / 2) * 2,
      H: Math.round(track.displayHeight * scale / 2) * 2,
      frames: [], tracks: [], perFrame: [], fixed: [],
    });
    view.width = v.W; view.height = v.H;
    video.src = URL.createObjectURL(file);
    $("vseek").max = duration;
    await analyze(track);
  } catch (e) {
    if (!(e instanceof UserError)) console.error(e);
    showError(e instanceof UserError ? e.message : `動画を読み込めませんでした。（${escapeHtml(e?.message || e)}）`);
  }
}
class UserError extends Error {}

/* Forget the previous video so nothing of it lingers after an error or cancel. */
function resetVideo() {
  video.pause();
  if (video.src) { URL.revokeObjectURL(video.src); video.removeAttribute("src"); video.load(); }
  Object.assign(v, { file: null, W: 0, H: 0, duration: 0, frames: [], tracks: [], perFrame: [], fixed: [] });
  v.thumbSeq++;
  regionCache = null;
  editor.selected = null;
  shownTime = 0;
  view.width = view.height = 0;
  $("vseek").value = 0;
  $("vtime").textContent = "0:00 / 0:00";
  renderPeople();
  updateButtons();
}
function showError(msg) {
  setProgress(null);
  setStatus(`<span class="err">${msg}</span>`);
  updateButtons();
}

/* ---------- Analysis ---------- */
async function analyze(track) {
  v.busy = true;
  updateButtons();
  keepAwake(true);
  let cancelled = false;
  v.cancel = () => { cancelled = true; };
  setStatus(detectorReady ? "顔を解析中…" : "AIモデルを読み込み中…", true);
  setProgress(0, "");
  try {
    const scale = Math.min(1, ANALYSIS_MAX_SIDE / Math.max(v.W, v.H));
    const aw = Math.round(v.W * scale), ah = Math.round(v.H * scale);
    const sink = new MB.CanvasSink(track, { width: aw, height: ah, fit: "fill", poolSize: 2 });
    const t0 = await track.getFirstTimestamp();
    const started = performance.now();
    let lastT = -Infinity, n = 0;
    let known = [];   // faces seen recently: { box, seen } in analysis pixels
    for await (const { canvas, timestamp } of sink.canvases()) {
      if (cancelled) break;
      const t = timestamp - t0;
      if (t - lastT < 1 / ANALYSIS_FPS - 0.002) continue;
      lastT = t;
      // Every frame: the whole frame plus a zoomed-in look around each known face
      // (keeps small faces tracked). Every DEEP_EVERY frames: finer tiles to find
      // new small faces — HOLD_SEC back-fills the short delay before one is found.
      const deep = n % DEEP_EVERY === 0;
      const raw = await detectRaw(canvas, { minConf: STORE_CONF, maxLevels: deep ? 2 : 0, yieldEvery: false });
      const rois = known.map(k => roiAround(k.box, aw, ah));
      if (rois.length) raw.push(...await detectInRects(canvas, rois, { minConf: STORE_CONF }));
      known = updateKnown(known, filterAndMerge(raw, FOLLOW_CONF), n);
      v.frames.push({ t, raw: raw.map(b => ({ x: b.x / aw, y: b.y / ah, w: b.w / aw, h: b.h / ah, score: b.score, tile: b.tile })) });
      if (++n % 5 === 0) {
        const frac = Math.min(1, t / v.duration);
        const elapsed = (performance.now() - started) / 1000;
        const left = frac > 0.05 ? Math.max(0, elapsed / frac - elapsed) : null;
        setStatus("顔を解析中…", true);
        setProgress(frac, `${Math.round(frac * 100)}%` + (left != null ? `・残り約${Math.ceil(left)}秒` : ""));
        await new Promise(r => setTimeout(r)); // let the UI update
      }
    }
    if (cancelled) {
      resetVideo();
      showError("解析を中止しました。別の動画を選ぶか、同じ動画をもう一度選んでください。");
      return;
    }
    if (!v.frames.length) throw new UserError("動画のフレームを読み込めませんでした。");
    build();
    setProgress(null);
    v.analysisSec = (performance.now() - started) / 1000;
    loadThumbs(track, t0);
  } finally {
    v.busy = false;
    v.cancel = null;
    keepAwake(false);
    updateButtons();
    showCount();
    drawPreview();
  }
}

/* Square area 3× the face size around it, clamped to the frame. */
function roiAround(b, W, H) {
  const s = Math.min(Math.max(b.w, b.h) * 3, W, H);
  const x = Math.min(Math.max(0, b.x + b.w / 2 - s / 2), W - s);
  const y = Math.min(Math.max(0, b.y + b.h / 2 - s / 2), H - s);
  return { x, y, w: s, h: s };
}

/* Refresh the list of faces to follow; forget ones not seen for FORGET_FRAMES. */
function updateKnown(known, found, n) {
  const next = known.filter(k => n - k.seen <= FORGET_FRAMES).map(k => ({ ...k }));
  for (const f of found) {
    const k = next.find(k => overlapRatio(k.box, f) > 0.3);
    if (k) { k.box = f; k.seen = n; } else next.push({ box: f, seen: n });
  }
  return next.slice(0, 12); // bound the per-frame cost for crowds
}

/* ---------- Tracking ---------- */
const iou = (a, b) => {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  return inter / Math.max(1e-9, a.w * a.h + b.w * b.h - inter);
};

/* Turns per-frame detections into per-person tracks and per-frame boxes.
   Re-run whenever the sensitivity changes (no need to decode the video again). */
function build() {
  const conf = sensitivityToConfidence($("vsensitivity").value);
  const N = v.frames.length;
  const fps = N / Math.max(0.001, v.frames[N - 1].t || v.duration);
  const gap = Math.max(1, Math.round(GAP_SEC * fps));
  const hold = Math.max(1, Math.round(HOLD_SEC * fps));
  const aspect = v.W / v.H;

  const tracks = [];
  for (let i = 0; i < N; i++) {
    const dets = filterAndMerge(v.frames[i].raw, conf).sort((a, b) => b.score - a.score);
    const used = new Set();
    for (const d of dets) {
      let best = null, bestScore = 0;
      for (const t of tracks) {
        const last = t.obs[t.obs.length - 1];
        if (used.has(t) || i - last.i > gap) continue;
        const b = last.box;
        const o = iou(d, b);
        // fall back to centre distance for fast movement (in pixel-proportional units)
        const dx = (d.x + d.w / 2 - b.x - b.w / 2) * aspect, dy = d.y + d.h / 2 - b.y - b.h / 2;
        const dist = Math.hypot(dx, dy) / Math.max(b.w * aspect, b.h);
        const sizeOk = d.w / b.w < 2.5 && b.w / d.w < 2.5;
        const s = o > 0.1 ? 1 + o : (sizeOk && dist < 1.2 ? 1 - dist / 1.2 : 0);
        if (s > bestScore) { bestScore = s; best = t; }
      }
      const box = { x: d.x, y: d.y, w: d.w, h: d.h };
      if (best) {
        best.obs.push({ i, box, score: d.score });
        used.add(best);
      } else {
        const t = { obs: [{ i, box, score: d.score }] };
        tracks.push(t);
        used.add(t);
      }
    }
  }
  // drop one-off blips that are not confident
  const kept = tracks.filter(t => t.obs.length >= 3 || Math.max(...t.obs.map(o => o.score)) >= conf + 0.15);

  // carry ON/OFF, labels and thumbnails over from the previous build where tracks match
  const old = v.tracks.slice();
  let nextLabel = 1;
  for (const t of old) nextLabel = Math.max(nextLabel, t.id + 1);
  kept.sort((a, b) => a.obs[0].i - b.obs[0].i);
  for (const t of kept) {
    const prev = matchOldTrack(t, old);
    if (prev) {
      Object.assign(t, { id: prev.id, label: prev.label, on: prev.on, thumb: prev.thumb });
      old.splice(old.indexOf(prev), 1);
    } else {
      t.id = nextLabel++;
      t.label = `人${t.id}`;
      t.on = true;
    }
    t.best = t.obs.reduce((a, b) => (b.score > a.score ? b : a));
  }

  // per-frame boxes: smoothed, gaps interpolated, held a little before/after
  const perFrame = Array.from({ length: N }, () => []);
  for (const t of kept) {
    const obs = smooth(t.obs);
    const put = (i, box) => { if (i >= 0 && i < N) perFrame[i].push({ track: t, ...box }); };
    for (let h = hold; h >= 1; h--) put(obs[0].i - h, obs[0].box);
    for (let k = 0; k < obs.length; k++) {
      const a = obs[k], b = obs[k + 1];
      put(a.i, a.box);
      if (!b) break;
      if (b.i - a.i <= gap) {
        for (let i = a.i + 1; i < b.i; i++) put(i, lerpBox(a.box, b.box, (i - a.i) / (b.i - a.i)));
      } else {
        for (let i = a.i + 1; i <= Math.min(a.i + hold, b.i - 1); i++) put(i, a.box);
        for (let i = Math.max(b.i - hold, a.i + hold + 1); i < b.i; i++) put(i, b.box);
      }
    }
    const last = obs[obs.length - 1];
    for (let h = 1; h <= hold; h++) put(last.i + h, last.box);
  }
  v.tracks = kept;
  v.perFrame = perFrame;
  regionCache = null;
  renderPeople();
}

function matchOldTrack(t, old) {
  let best = null, bestHits = 0;
  for (const o of old) {
    const byFrame = new Map(o.obs.map(x => [x.i, x.box]));
    let hits = 0;
    for (const x of t.obs) {
      const b = byFrame.get(x.i);
      if (b && iou(b, x.box) > 0.3) hits++;
    }
    if (hits > bestHits) { bestHits = hits; best = o; }
  }
  return bestHits >= Math.min(3, t.obs.length) ? best : null;
}

const lerpBox = (a, b, f) => ({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, w: a.w + (b.w - a.w) * f, h: a.h + (b.h - a.h) * f });

/* Light moving average over neighbouring observations to reduce jitter. */
function smooth(obs) {
  return obs.map((o, k) => {
    const near = obs.slice(Math.max(0, k - 2), k + 3).filter(x => Math.abs(x.i - o.i) <= 3);
    const avg = (key) => near.reduce((s, x) => s + x.box[key], 0) / near.length;
    return { i: o.i, box: { x: avg("x"), y: avg("y"), w: avg("w"), h: avg("h") } };
  });
}

/* ---------- Regions at a given time ---------- */
function frameIndexAt(t) {
  const f = v.frames;
  if (!f.length) return -1;
  let lo = 0, hi = f.length - 1;
  const half = f.length > 1 ? (f[1].t - f[0].t) / 2 : 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (f[mid].t <= t + half) lo = mid; else hi = mid - 1;
  }
  return lo;
}

let regionCache = null;
function trackRegionsAt(i) {
  const pad = Number($("vpadding").value) + MOTION_MARGIN;
  if (regionCache && regionCache.i === i && regionCache.pad === pad) return regionCache.list;
  const list = (v.perFrame[i] || []).map(b => ({
    ...padBox({ x: b.x * v.W, y: b.y * v.H, w: b.w * v.W, h: b.h * v.H }, pad, v.W, v.H),
    track: b.track, on: b.track.on, label: b.track.label, editable: false,
  }));
  regionCache = { i, pad, list };
  return list;
}
const regionsAt = (t) => [...trackRegionsAt(frameIndexAt(t)), ...v.fixed];

/* ---------- Preview ---------- */
const editor = createRegionEditor(view, {
  getRegions: () => (v.frames.length ? regionsAt(shownTime) : []),
  onToggle: (r) => {
    if (r.track) r.track.on = !r.track.on; else r.on = !r.on;
    regionCache = null;
    renderPeople();
  },
  onAdd: (rect) => {
    const r = { ...rect, on: true, label: "固定" };
    v.fixed.push(r);
    return r;
  },
  onDelete: (r) => { v.fixed.splice(v.fixed.indexOf(r), 1); },
  onChange: () => { updateButtons(); drawPreview(); showCount(); },
});

// mediaTime of the frame currently shown by <video> (more exact than currentTime)
let shownTime = 0;
function drawPreview() {
  if (!v.W || video.readyState < 2) return;
  ctx.drawImage(video, 0, 0, v.W, v.H);
  if (!v.frames.length) return;
  const regions = regionsAt(shownTime);
  const opts = effectOpts();
  for (const r of regions) if (r.on) applyEffect(ctx, view, r, opts);
  if ($("vshowFrames").checked) drawFrames(ctx, view, regions, editor.selected, editor.newRect());
}

// Redraw whenever <video> presents a new frame (playback and seeks). 'seeked'
// fires before the new frame is on screen, so drawing there shows the old one.
if (video.requestVideoFrameCallback) {
  let handle = 0;
  const onFrame = (_now, meta) => {
    shownTime = meta.mediaTime;
    drawPreview();
    handle = video.requestVideoFrameCallback(onFrame);
  };
  // (re)arm for every newly loaded video: a source change may drop a pending callback
  const arm = () => {
    if (handle) video.cancelVideoFrameCallback(handle);
    handle = video.requestVideoFrameCallback(onFrame);
    drawPreview();
  };
  video.addEventListener("loadeddata", arm);
  arm();
} else {
  const loop = () => {
    shownTime = video.currentTime;
    drawPreview();
    if (!video.paused && !video.ended) requestAnimationFrame(loop);
  };
  video.addEventListener("play", loop);
  video.addEventListener("seeked", () => requestAnimationFrame(loop));
  video.addEventListener("loadeddata", loop);
}
video.addEventListener("play", () => { $("vplay").textContent = "❚❚ 一時停止"; });
video.addEventListener("pause", () => { $("vplay").textContent = "▶ 再生"; });
video.addEventListener("timeupdate", () => {
  $("vseek").value = video.currentTime;
  $("vtime").textContent = `${fmtTime(video.currentTime)} / ${fmtTime(v.duration)}`;
});
$("vplay").addEventListener("click", () => (video.paused ? video.play() : video.pause()));
$("vseek").addEventListener("input", (e) => { video.currentTime = Number(e.target.value); });

/* ---------- People list ---------- */
function renderPeople() {
  const box = $("vpeople");
  if (!v.tracks.length) {
    box.innerHTML = `<p class="muted-note">${v.frames.length ? "顔が見つかりませんでした。" : "解析後に表示されます。"}</p>`;
    return;
  }
  box.innerHTML = "";
  for (const t of v.tracks) {
    const row = document.createElement("div");
    row.className = "person" + (t.on ? "" : " off");
    const first = v.frames[t.obs[0].i].t, last = v.frames[t.obs[t.obs.length - 1].i].t;
    row.innerHTML = `
      <button class="thumb" title="この人が映る場面へ移動">${t.thumb ? `<img src="${t.thumb}" alt="">` : ""}</button>
      <div class="pinfo"><b>${t.label}</b><span>${fmtTime(first)}〜${fmtTime(last)}</span></div>
      <label class="switch"><input type="checkbox" ${t.on ? "checked" : ""}><span>${t.on ? "ON" : "OFF"}</span></label>`;
    row.querySelector("input").addEventListener("change", (e) => {
      t.on = e.target.checked;
      regionCache = null;
      renderPeople();
      drawPreview();
      showCount();
    });
    row.querySelector(".thumb").addEventListener("click", () => {
      video.pause();
      video.currentTime = v.frames[t.best.i].t;
    });
    box.appendChild(row);
  }
}

/* Face thumbnails for the people list (decoded once per person). */
async function loadThumbs(track, t0) {
  const seq = ++v.thumbSeq;
  const sink = new MB.CanvasSink(track, { width: v.W, height: v.H, fit: "fill" });
  for (const t of v.tracks) {
    if (t.thumb) continue;
    const w = await sink.getCanvas(v.frames[t.best.i].t + t0);
    if (seq !== v.thumbSeq) return;
    if (!w) continue;
    const b = t.best.box;
    const r = padBox({ x: b.x * v.W, y: b.y * v.H, w: b.w * v.W, h: b.h * v.H }, 30, v.W, v.H);
    const c = document.createElement("canvas");
    c.width = c.height = 96;
    const s = Math.max(r.w, r.h);
    c.getContext("2d").drawImage(w.canvas, r.x + r.w / 2 - s / 2, r.y + r.h / 2 - s / 2, s, s, 0, 0, 96, 96);
    t.thumb = c.toDataURL("image/jpeg", 0.8);
    renderPeople();
  }
}

function showCount() {
  if (v.busy || !v.frames.length) return;
  const n = v.tracks.length, on = v.tracks.filter(t => t.on).length + v.fixed.filter(r => r.on).length;
  setStatus(n
    ? `<b>${n}人</b>を検出しました（モザイク ${on}か所）・再生して確認してください`
    : `顔が見つかりませんでした。感度を上げるか、ドラッグで固定の範囲を追加してください。`);
}

function updateButtons() {
  const ready = !!v.frames.length && !v.busy;
  $("vexport").disabled = !ready;
  $("vcancel").hidden = !v.busy;
  $("vdelSel").disabled = !editor.canDelete();
  $("vanother").disabled = v.busy;
  $("vallOn").disabled = $("vallOff").disabled = !ready;
}

/* ---------- Export ---------- */
async function exportVideo() {
  if (!v.frames.length || v.busy) return;
  video.pause();
  v.busy = true;
  updateButtons();
  keepAwake(true);
  setStatus("動画を書き出し中…（画面を閉じないでください）", true);
  setProgress(0, "0%");
  let conversion = null;
  v.cancel = () => conversion?.cancel();
  try {
    const input = new MB.Input({ source: new MB.BlobSource(v.file), formats: MB.ALL_FORMATS });
    const output = new MB.Output({ format: new MB.Mp4OutputFormat({ fastStart: "in-memory" }), target: new MB.BufferTarget() });
    const codec = (await MB.canEncodeVideo("avc", { width: v.W, height: v.H })) ? "avc" : undefined;
    const work = document.createElement("canvas");
    work.width = v.W; work.height = v.H;
    const wctx = work.getContext("2d");
    const opts = effectOpts();
    let first = null;

    conversion = await MB.Conversion.init({
      input, output, tracks: "primary", showWarnings: false,
      video: {
        allowTransformationMetadata: false, // bake rotation into the pixels so boxes line up
        forceTranscode: true,
        codec,
        quality: MB.QUALITY_HIGH,
        processedWidth: v.W,
        processedHeight: v.H,
        process: (sample) => {
          if (first === null) first = sample.timestamp;
          sample.draw(wctx, 0, 0, v.W, v.H);
          const t = sample.timestamp - first;
          for (const r of [...trackRegionsAt(frameIndexAt(t)), ...v.fixed]) if (r.on) applyEffect(wctx, work, r, opts);
          return new MB.VideoSample(work, { timestamp: sample.timestamp, duration: sample.duration });
        },
      },
    });
    if (!conversion.isValid) throw new UserError("この端末ではこの動画を書き出せませんでした。");
    const audioLost = conversion.discardedTracks.some(d => d.track.type === "audio");
    conversion.onProgress = (p) => setProgress(p, `${Math.round(p * 100)}%`);
    await conversion.execute();

    const blob = new Blob([output.target.buffer], { type: "video/mp4" });
    const base = v.file.name.replace(/\.[^.]+$/, "");
    downloadBlob(blob, `${base}_mosaic.mp4`);
    setProgress(null);
    setStatus(`書き出しが完了しました（${(blob.size / 1048576).toFixed(1)}MB）` +
      (audioLost ? '・<span class="err">この動画の音声形式には対応しておらず、音声なしで書き出しました</span>' : ""));
  } catch (e) {
    console.error(e);
    if (e?.name === "ConversionCanceledError" || conversion?.state === "canceled") showError("書き出しを中止しました。");
    else showError(e instanceof UserError ? e.message : `書き出しに失敗しました。（${escapeHtml(e?.message || e)}）`);
  } finally {
    v.busy = false;
    v.cancel = null;
    keepAwake(false);
    updateButtons();
  }
}

/* ---------- Controls ---------- */
$("vdrop").addEventListener("click", () => $("vfile").click());
$("vdrop").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("vfile").click(); } });
$("vfile").addEventListener("change", (e) => { loadVideoFile(e.target.files[0]); e.target.value = ""; });
$("vanother").addEventListener("click", () => $("vfile").click());
$("vexport").addEventListener("click", exportVideo);
$("vcancel").addEventListener("click", () => v.cancel?.());
$("vdelSel").addEventListener("click", () => editor.deleteSelected());
$("vallOn").addEventListener("click", () => setAll(true));
$("vallOff").addEventListener("click", () => setAll(false));
function setAll(on) {
  v.tracks.forEach(t => t.on = on);
  v.fixed.forEach(r => r.on = on);
  regionCache = null;
  renderPeople();
  drawPreview();
  showCount();
}

$("vstrength").addEventListener("input", (e) => { $("vstrengthOut").textContent = e.target.value; drawPreview(); });
$("vpadding").addEventListener("input", (e) => { $("vpaddingOut").textContent = e.target.value + "%"; regionCache = null; drawPreview(); });
let sensTimer;
$("vsensitivity").addEventListener("input", (e) => {
  $("vsensitivityOut").textContent = e.target.value;
  clearTimeout(sensTimer);
  sensTimer = setTimeout(() => {
    if (!v.frames.length || v.busy) return;
    build();
    showCount();
    drawPreview();
  }, 250);
});
document.querySelectorAll("input[name=veffect], input[name=vshape]").forEach(el => el.addEventListener("change", drawPreview));
$("vshowFrames").addEventListener("change", drawPreview);
window.addEventListener("resize", drawPreview);

export function deleteSelectedVideoRegion() { return editor.deleteSelected(); }
export const videoBusy = () => v.busy;
renderPeople();
