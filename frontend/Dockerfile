# syntax=docker/dockerfile:1

# =========================================================
# Frontend image — static UI served by nginx, /api proxied
# =========================================================
FROM nginx:alpine

# Site config: serve the static UI and reverse-proxy /api to the backend.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Static assets.
COPY index.html style.css app.js /usr/share/nginx/html/

# In the container the UI talks to the backend same-origin (nginx proxies
# /api), so ship the Docker runtime config as config.js.
COPY config.docker.js /usr/share/nginx/html/config.js

EXPOSE 80
