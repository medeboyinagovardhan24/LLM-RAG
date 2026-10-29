
import functools
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

BACKEND_HOST = "127.0.0.1"
FRONTEND_HOST = "127.0.0.1"


def find_free_port(start_port: int, host: str = "127.0.0.1", max_attempts: int = 30) -> int:
    """Return the first free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting from {start_port} on {host}.")


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if a port is available to bind right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def choose_unique_ports(backend_start: int, frontend_start: int):
    """Select distinct free backend/frontend ports so the UI and API do not collide."""
    backend_port = find_free_port(backend_start)
    frontend_port = frontend_start

    while frontend_port == backend_port or not is_port_free(frontend_port):
        frontend_port += 1

    return backend_port, frontend_port


BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8010))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 5888))


def serve_frontend(port: int, backend_port: int):
    """Serve the frontend and provide its selected API URL without editing files."""
    class FrontendHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] == "/config.js":
                config = json.dumps({"apiBase": f"http://{BACKEND_HOST}:{backend_port}"})
                body = f"window.__APP_CONFIG__ = {config};\n".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    handler = functools.partial(
        FrontendHandler, directory=str(FRONTEND)
    )
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((FRONTEND_HOST, port), handler) as httpd:
        httpd.serve_forever()


def main():
    global BACKEND_PORT, FRONTEND_PORT

    BACKEND_PORT, FRONTEND_PORT = choose_unique_ports(BACKEND_PORT, FRONTEND_PORT)

    # Make the backend's `app` package importable (backend/app/...).
    sys.path.insert(0, str(BACKEND))

    try:
        import uvicorn
    except ModuleNotFoundError:
        sys.exit(
            "\n[!] uvicorn is not installed in the interpreter you're using.\n"
            "    In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter'\n"
            "    -> choose the interpreter inside the .venv folder, then Run again.\n"
        )

    threading.Thread(
        target=serve_frontend,
        args=(FRONTEND_PORT, BACKEND_PORT),
        daemon=True,
    ).start()

    print("=" * 62)
    print("  LLM-RAG: Domain-Specific Question Answering System is starting...")
    print(f"  ->  OPEN THIS IN YOUR BROWSER:  http://{FRONTEND_HOST}:{FRONTEND_PORT}")
    print(f"  ->  API docs (Swagger):         http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print("  ->  Stop: press Ctrl+C or click the stop icon.")
    print("=" * 62)

    uvicorn.run(
        "app.main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
