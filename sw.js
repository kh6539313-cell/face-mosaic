// Service Worker: caches the whole app (including the AI model) for offline use.
// Bump VERSION whenever any file below changes so clients pick up the update.
const VERSION = "v3";
const CACHE = `face-mosaic-${VERSION}`;
const FILES = [
  "./",
  "index.html",
  "manifest.json",
  "icons/icon-180.png",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/icon-512-maskable.png",
  "vendor/vision_bundle.mjs",
  "vendor/blaze_face_short_range.tflite",
  "vendor/wasm/vision_wasm_internal.js",
  "vendor/wasm/vision_wasm_internal.wasm",
  "vendor/wasm/vision_wasm_nosimd_internal.js",
  "vendor/wasm/vision_wasm_nosimd_internal.wasm",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("face-mosaic-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  // The page itself: network first so fixes show up immediately when online.
  if (e.request.mode === "navigate") {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put("index.html", copy));
          return res;
        })
        .catch(() => caches.match("index.html"))
    );
    return;
  }
  // Everything else (library, model, icons): cache first.
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request))
  );
});
