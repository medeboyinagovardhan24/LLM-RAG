/* =========================================================
   RUNTIME CONFIGURATION — Docker / container build
   ---------------------------------------------------------
   Empty apiBase means "same origin". nginx (see nginx.conf)
   serves this UI and reverse-proxies /api to the backend,
   so the browser calls /api/... on its own origin.
========================================================= */

window.__APP_CONFIG__ = {
    apiBase: ""
};
