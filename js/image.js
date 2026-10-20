// Image tool: detect faces in a still image, adjust regions, save.
import {
  $, getDetector, detectorReady, detectRaw, filterAndMerge, sensitivityToConfidence,
  padBox, overlapRatio, applyEffect, drawFrames, createRegionEditor, downloadBlob, escapeHtml,
} from "./core.js";

const view = $("view");
const ctx = view.getContext("2d");

const state = {
  img: null,          // HTMLCanvasElement holding the original pixels
  fileName: "image",
  faces: [],          // detected boxes (before padding) { x, y, w, h }
  regions: [],        // { x, y, w, h, on, auto, faceIndex? }
};

function setStatus(html, busy = false) {
  $("statusText").innerHTML = html;
  $("status").classList.toggle("busy", busy);
}

const confidence = () => sensitivityToConfidence($("sensitivity").value);
const padFace = (f) => padBox(f, Number($("padding").value), state.img.width, state.img.height);
const effectOpts = () => ({
  effect: document.querySelector("input[name=effect]:checked").value,
  shape: document.querySelector("input[name=shape]:checked").value,
  strength: Number($("strength").value),
});

const editor = createRegionEditor(view, {
  getRegions: () => state.regions,
  onToggle: (r) => { r.on = !r.on; },
  onAdd: (rect) => {
    const r = { ...rect, on: true, auto: false };
    state.regions.push(r);
    return r;
  },
  // once adjusted by hand, padding changes / re-detection leave it alone
  onEdit: (r) => { r.auto = false; },
  onDelete: (r) => { state.regions.splice(state.regions.indexOf(r), 1); },
  onChange: () => { updateButtons(); render(); showCount(); },
});

/* ---------- Loading ---------- */
export async function loadImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus('<span class="err">画像ファイルを選んでください。</span>');
    return;
  }
  try {
    // imageOrientation "from-image" applies the EXIF rotation of phone photos.
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const c = document.createElement("canvas");
    c.width = bmp.width; c.height = bmp.height;
    c.getContext("2d").drawImage(bmp, 0, 0);
    bmp.close();
    state.img = c;
    state.fileName = file.name || "image";
  } catch (e) {
    setStatus('<span class="err">画像を読み込めませんでした。</span>');
    return;
  }
  view.width = state.img.width;
  view.height = state.img.height;
  state.faces = [];
  state.regions = [];
  editor.selected = null;
  $("drop").style.display = "none";
  $("work").classList.add("show");
  render();
  await redetect();
}

let detectSeq = 0;
async function redetect() {
  if (!state.img) return;
  const seq = ++detectSeq;
  setStatus(detectorReady ? "顔を検出中…" : "AIモデルを読み込み中…（初回のみ数秒かかります）", true);
  try {
    const conf = confidence();
    const faces = filterAndMerge(await detectRaw(state.img, { minConf: conf }), conf);
    if (seq !== detectSeq) return;
    state.faces = faces;
    // Keep manual / hand-adjusted regions, and skip detections they already cover.
    const manual = state.regions.filter(r => !r.auto);
    state.regions = [
      ...faces.map((f, i) => ({ ...padFace(f), on: true, auto: true, faceIndex: i }))
        .filter(r => !manual.some(m => overlapRatio(r, m) > 0.5)),
      ...manual,
    ];
    editor.selected = null;
    updateButtons();
    render();
    showCount();
  } catch (e) {
    console.error(e);
    if (seq === detectSeq) {
      setStatus(`<span class="err">顔検出に失敗しました。ページを再読み込みしてください。（${escapeHtml(e?.message || e)}）</span>`);
    }
  }
}

function showCount() {
  if (!state.img) return;
  const n = state.faces.length;
  const on = state.regions.filter(r => r.on).length;
  setStatus(n
    ? `<b>${n}人</b>の顔を検出しました（モザイク ${on}か所）`
    : `顔が見つかりませんでした。感度を上げるか、ドラッグで範囲を追加してください。（モザイク ${on}か所）`);
}

/* ---------- Rendering ---------- */
function paint(g, withFrames) {
  const src = state.img;
  g.imageSmoothingEnabled = true;
  g.drawImage(src, 0, 0);
  const opts = effectOpts();
  for (const r of state.regions) if (r.on) applyEffect(g, src, r, opts);
  if (withFrames) drawFrames(g, view, state.regions, editor.selected, editor.newRect());
}

let rafPending = false;
function render() {
  if (!state.img || rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    paint(ctx, $("showFrames").checked);
  });
}

function updateButtons() {
  $("delSel").disabled = !editor.canDelete();
}

/* ---------- Controls ---------- */
$("drop").addEventListener("click", () => $("file").click());
$("drop").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
$("file").addEventListener("change", (e) => { loadImageFile(e.target.files[0]); e.target.value = ""; });
$("another").addEventListener("click", () => $("file").click());

export function deleteSelectedImageRegion() { return editor.deleteSelected(); }

$("strength").addEventListener("input", (e) => { $("strengthOut").textContent = e.target.value; render(); });
$("padding").addEventListener("input", (e) => {
  $("paddingOut").textContent = e.target.value + "%";
  if (!state.img) return;
  for (const r of state.regions) if (r.auto) Object.assign(r, padFace(state.faces[r.faceIndex]));
  render();
});
let sensTimer;
$("sensitivity").addEventListener("input", (e) => {
  $("sensitivityOut").textContent = e.target.value;
  clearTimeout(sensTimer);
  sensTimer = setTimeout(redetect, 350);
});
document.querySelectorAll("input[name=effect], input[name=shape]").forEach(el => el.addEventListener("change", render));
$("showFrames").addEventListener("change", render);
$("delSel").addEventListener("click", () => editor.deleteSelected());
$("allOn").addEventListener("click", () => { state.regions.forEach(r => r.on = true); render(); showCount(); });
$("allOff").addEventListener("click", () => { state.regions.forEach(r => r.on = false); render(); showCount(); });
window.addEventListener("resize", render);

$("save").addEventListener("click", () => {
  if (!state.img) return;
  const fmt = document.querySelector("input[name=fmt]:checked").value;
  const out = document.createElement("canvas");
  out.width = state.img.width; out.height = state.img.height;
  paint(out.getContext("2d"), false);
  out.toBlob((blob) => {
    const base = state.fileName.replace(/\.[^.]+$/, "");
    downloadBlob(blob, `${base}_mosaic.${fmt === "jpeg" ? "jpg" : "png"}`);
  }, `image/${fmt}`, 0.92);
});

// Warm up the model in the background so the first detection is faster.
getDetector().catch(() => {});
