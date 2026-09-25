#!/usr/bin/env python3
"""Tiny local server for the Fly Brain Explorer.

- Serves the static web app from this directory.
- Proxies neuroglancer skeleton blobs from the public HHMI Janelia
  'flyem-male-cns' GCS bucket (the browser cannot fetch them directly
  because the bucket sends no CORS headers), caching them under data/skeletons/.

Usage:  python server.py  [port]     (default port 8000)
"""
import http.server
import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
SKEL_BASE = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
             "skeletons-malecns-mirrored/skeletons-precomputed/")
CACHE = os.path.join(ROOT, "data", "skeletons")


class Handler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        if self.path.startswith("/skeleton/"):
            self._proxy_skeleton()
        else:
            super().do_GET()

    def _proxy_skeleton(self):
        body_id = self.path.rsplit("/", 1)[-1].split("?")[0]
        if not body_id.isdigit():
            self.send_error(400, "bad bodyId")
            return
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, body_id)
        try:
            with open(path, "rb") as f:
                data = f.read()
            status = "cached"
        except OSError:
            url = SKEL_BASE + body_id
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                shutil.move(tmp, path)
                status = "fetched"
            except Exception as exc:  # noqa: BLE001
                self.send_error(502, f"upstream error: {exc}")
                return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        sys.stderr.write(f"[skeleton {body_id}] {status} ({len(data)} bytes)\n")

    def log_message(self, fmt, *args):  # quiet default logging
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.makedirs(CACHE, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Fly Brain Explorer running at http://127.0.0.1:{port}/")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
