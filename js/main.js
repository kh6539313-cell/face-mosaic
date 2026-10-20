// Tabs, file drop / paste routing, keyboard shortcuts, offline support.
import { $ } from "./core.js";
import { loadImageFile, deleteSelectedImageRegion } from "./image.js";
import { loadVideoFile, deleteSelectedVideoRegion } from "./video.js";

let active = "image";
function showTab(name) {
  active = name;
  for (const t of ["image", "video"]) {
    $(`tab-${t}`).setAttribute("aria-selected", String(t === name));
    $(`panel-${t}`).hidden = t !== name;
  }
  try { localStorage.setItem("tab", name); } catch { /* storage may be unavailable */ }
}
$("tab-image").addEventListener("click", () => showTab("image"));
$("tab-video").addEventListener("click", () => showTab("video"));
try { if (localStorage.getItem("tab") === "video") showTab("video"); } catch { /* ignore */ }

const isVideo = (f) => f.type.startsWith("video/") || /\.(mp4|mov|m4v|webm|mkv)$/i.test(f.name);

// Drop anywhere: images go to the image tab, videos to the video tab.
["dragenter", "dragover"].forEach(t => document.addEventListener(t, (e) => {
  e.preventDefault();
  $(active === "image" ? "drop" : "vdrop").classList.add("over");
}));
["dragleave", "drop"].forEach(t => document.addEventListener(t, (e) => {
  e.preventDefault();
  if (t === "dragleave" && e.relatedTarget) return;
  $("drop").classList.remove("over");
  $("vdrop").classList.remove("over");
}));
document.addEventListener("drop", (e) => {
  const f = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith("image/") || isVideo(f));
  if (!f) return;
  if (isVideo(f)) { showTab("video"); loadVideoFile(f); }
  else { showTab("image"); loadImageFile(f); }
});
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
  if (item) { showTab("image"); loadImageFile(item.getAsFile()); }
});
document.addEventListener("keydown", (e) => {
  if ((e.key === "Delete" || e.key === "Backspace") && !e.target.matches?.("textarea, input[type=text]")) {
    const done = active === "image" ? deleteSelectedImageRegion() : deleteSelectedVideoRegion();
    if (done) e.preventDefault();
  }
});

// Offline support: cache the app and model after the first visit.
if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("SW registration failed", e));
}
