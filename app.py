"""Web server. Serves the page and three endpoints: status, ingest, ask.

Run it with: python3 app.py
"""

import json
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rag

PORT = int(os.environ.get("PORT", 8000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):

    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split("?")[0]          # drop ?q=... if there is one
        if path in ("/", "/index.html"):
            with open(os.path.join(BASE_DIR, "static", "index.html"), "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif path == "/api/status":
            info = rag.stats()
            info["model"] = rag.foundry_model()
            info["endpoint"] = rag.foundry_endpoint()
            self._send(200, info)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/api/ingest":
            self._send(200, rag.ingest())

        elif self.path == "/api/ask":
            question = (body.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "Question is empty."})
                return
            started = time.time()
            try:
                result = rag.answer_query(question)
            except Exception as exc:                      # show the reason on the page
                self._send(500, {"error": str(exc)})
                return
            result["seconds"] = round(time.time() - started, 2)
            self._send(200, result)

        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):                          # stop it logging every request
        pass


if __name__ == "__main__":
    if rag.stats()["chunks"] == 0:
        print("Building the index from documents/ ...")
        print("  ->", rag.ingest())

    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print("Port {} is already in use. Close the other server or run: "
              "PORT=8001 python3 app.py".format(PORT))
        raise SystemExit(1)

    print("Local RAG Assistant running at http://localhost:{}".format(PORT))
    print("Press Ctrl+C to stop.")
    webbrowser.open("http://localhost:{}".format(PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
