// PropertyHUB — Service Worker
// مطلوب لإظهار prompt "Install / Add to Home Screen"

const CACHE_NAME = 'propertyhub-v1';

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

// ✅ تجاهل تماماً أي request مش على نفس origin الموقع (يعني الـ API على 8000)
// عشان السيرفيس وركر مايتدخلش في requests الـ backend ويسبب CORS errors
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (url.origin !== self.location.origin) {
    return; // سيب الـ browser يتعامل معاها عادي
  }

  if (event.request.method !== 'GET') {
    return; // متلمسش POST/PUT/DELETE
  }

  event.respondWith(
    fetch(event.request)
      .then((res) => res)
      .catch(() =>
        caches.match(event.request).then((cached) => cached || Response.error())
      )
  );
});