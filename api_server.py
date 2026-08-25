"""
SpaceaxAI - REST API & Web UI Server v3.0
Server HTTP multi-thread yang menyediakan:
1. REST API OpenAI-compatible (/v1/chat/completions, /v1/models, /api/agent)
2. Static Web Server untuk ChatGPT-style Web UI (web/)
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import os
import sys
import json
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading

# Tambahkan root folder ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import get_config, AI_IDENTITY
from chat import get_fallback, is_valid_output
from core.agent import SpaceaxAgent


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Server HTTP multi-thread agar tidak memblokir antrean request web."""
    daemon_threads = True


class SpaceaxAPIHandler(BaseHTTPRequestHandler):
    """Handler HTTP untuk API OpenAI-Compatible + Interface Web ChatGPT-Style."""

    def log_message(self, format, *args):
        """Custom silent/clean logging."""
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]}\n")

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        path = self.path.split("?")[0]

        # 1. Models List (OpenAI compatible)
        if path == "/v1/models":
            self._set_headers(200)
            models = {
                "object": "list",
                "data": [
                    {"id": "spaceax-small", "object": "model", "owned_by": "spaceax"},
                    {"id": "spaceax-medium", "object": "model", "owned_by": "spaceax"},
                    {"id": "spaceax-promax-8b", "object": "model", "owned_by": "spaceax"},
                ]
            }
            self.wfile.write(json.dumps(models).encode("utf-8"))
            return

        # 2. System Status
        if path == "/api/status":
            self._set_headers(200)
            status = {
                "status": "online",
                "identity": AI_IDENTITY,
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        # 3. Serve Web UI Static Files (web/index.html, app.js, style.css)
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        if path == "/" or path == "/index.html":
            file_path = os.path.join(web_dir, "index.html")
            content_type = "text/html; charset=utf-8"
        else:
            file_path = os.path.join(web_dir, path.lstrip("/"))
            if path.endswith(".css"):
                content_type = "text/css"
            elif path.endswith(".js"):
                content_type = "application/javascript"
            elif path.endswith(".png"):
                content_type = "image/png"
            elif path.endswith(".jpg") or path.endswith(".jpeg"):
                content_type = "image/jpeg"
            else:
                content_type = "text/plain"

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self._set_headers(200, content_type)
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._set_headers(404, "text/html")
            self.wfile.write(b"<h1>404 Not Found</h1>")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body = {}

        path = self.path.split("?")[0]

        # 1. Chat Completions API (OpenAI Compatible)
        if path in ["/v1/chat/completions", "/api/chat"]:
            messages = body.get("messages", [])
            web_enabled = body.get("web_enabled", True)
            user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break

            if not user_msg:
                user_msg = body.get("prompt", "Halo")

            # Emulasi balasan kecerdasan SpaceAX
            resp_text = get_fallback(user_msg)
            if not resp_text:
                resp_text = f"Pertanyaan Anda mengenai '{user_msg}' telah diproses."

            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "spaceax-small"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": resp_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_msg.split()),
                    "completion_tokens": len(resp_text.split()),
                    "total_tokens": len(user_msg.split()) + len(resp_text.split())
                }
            }

            self._set_headers(200)
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            return

        # 2. Agent Execution API
        if path == "/api/agent":
            goal = body.get("goal", body.get("prompt", ""))
            agent = SpaceaxAgent()
            res = agent.run_agent_loop(goal)

            self._set_headers(200)
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        self._set_headers(400)
        self.wfile.write(json.dumps({"error": "Unknown endpoint"}).encode("utf-8"))


def start_server(host: str = "0.0.0.0", port: int = 7860, size_override: str = None):
    """Jalankan server API dan Web UI."""
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, SpaceaxAPIHandler)
    
    cfg = get_config(auto_detect=True, size_override=size_override)
    active_profile = cfg.get("promax_tier") or cfg.get("profile_name", "auto")

    print("=" * 65)
    print("SpaceAX AI v3.0 | Server REST API & Web UI Service")
    print("=" * 65)
    print(f"Active Model Profile: {active_profile.upper()}")
    print(f"Web UI:               http://localhost:{port}")
    print(f"API Base:             http://localhost:{port}/v1")
    print(f"Models List:          http://localhost:{port}/v1/models")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer terminated.")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
