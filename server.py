#!/usr/bin/env python3
"""Tiny local server for the Fly Brain Explorer.

- FIRST RUN: downloads the open-source MaleCNS v1.0 connectome dataset
  (HHMI Janelia, CC-BY-4.0) from the public 'flyem-male-cns' GCS bucket
  if it is not already in data/ — with resume support and progress, so an
  interrupted download continues where it stopped. It then runs the
  distillation scripts (prepare_data.py / prepare_olfactory.py) if their
  outputs are missing, so a fresh checkout fully sets itself up.
- Serves the static web app from this directory.
- Proxies neuroglancer skeleton blobs from the same public bucket (the
  browser cannot fetch them directly because the bucket sends no CORS
  headers), caching them under data/skeletons/.

Usage:  python server.py  [port]     (default port 8000)
        python server.py --no-download   (serve only, never download)
"""
import http.server
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
SCRIPTS = os.path.join(ROOT, "scripts")
CACHE = os.path.join(DATA, "skeletons")
SKEL_BASE = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
             "skeletons-malecns-mirrored/skeletons-precomputed/")

# raw dataset files: local name -> (bucket path, bytes for progress display)
DATASET_FILES = {
    "body-annotations.feather": (
        "v1.0/connectome-data/flat-connectome/"
        "body-annotations-male-cns-v1.0-minconf-0.5.feather", 14_483_314),
    "body-neurotransmitters.feather": (
        "v1.0/connectome-data/flat-connectome/"
        "body-neurotransmitters-male-cns-v1.0.feather", 43_282_834),
    "connectome-weights.feather": (
        "v1.0/connectome-data/flat-connectome/"
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1_051_241_946),
}

# derived files and the script that (re)builds them from the raw dataset
DERIVED = [
    ("neurons.tsv.gz", "prepare_data.py"),
    ("edges_pre.u32", "prepare_data.py"),
    ("edges_post.u32", "prepare_data.py"),
    ("edges_w.u16", "prepare_data.py"),
    ("edges_idx_ids.u32", "prepare_data.py"),
    ("edges_idx_off.u32", "prepare_data.py"),
    ("redges_pre.u32", "prepare_data.py"),
    ("redges_w.u16", "prepare_data.py"),
    ("redges_idx_ids.u32", "prepare_data.py"),
    ("redges_idx_off.u32", "prepare_data.py"),
    ("meta.json", "prepare_data.py"),
    ("olfactory_circuit.json", "prepare_olfactory.py"),
]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0


def need_download():
    return any(
        not os.path.exists(os.path.join(DATA, name)) or
        os.path.getsize(os.path.join(DATA, name)) < expected * 0.98
        for name, (_, expected) in DATASET_FILES.items()
    )


def need_derive():
    return any(not os.path.exists(os.path.join(DATA, name))
               for name, _ in DERIVED)


def download_file(local, bucket_path, expected):
    """Stream one dataset file with resume + progress. Returns True on success."""
    dest = os.path.join(DATA, local)
    tmp = dest + ".part"
    os.makedirs(DATA, exist_ok=True)
    url = "https://storage.googleapis.com/flyem-male-cns/" + bucket_path
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    req = urllib.request.Request(url)
    if start > 0:
        req.add_header("Range", f"bytes={start}-")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            if start > 0 and r.status != 206:
                # server ignored the Range header — restart from scratch
                start = 0
            total = expected
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                try:
                    total = int(cr.split("/")[-1])
                except ValueError:
                    pass
            t0 = time.time()
            got = 0
            with open(tmp, "ab" if start > 0 else "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got % (32 << 20) < (1 << 20):
                        rate = got / max(1e-9, time.time() - t0)
                        pct = (start + got) / max(1, total) * 100
                        print(f"  {local}: {human(start + got)}/{human(total)} "
                              f"({pct:.0f}%) {human(rate)}/s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! download failed: {exc}", flush=True)
        return False

    size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if size < expected * 0.98:
        print(f"  ! {local}: incomplete ({human(size)}/{human(expected)})", flush=True)
        return False
    shutil.move(tmp, dest)
    return True


def download_dataset():
    print("=" * 62)
    print("First run: downloading the MaleCNS v1.0 connectome dataset")
    print("(HHMI Janelia, CC-BY-4.0, public bucket — about 1.1 GB total)")
    print("=" * 62)
    ok = True
    for local, (bucket_path, expected) in DATASET_FILES.items():
        dest = os.path.join(DATA, local)
        if os.path.exists(dest) and os.path.getsize(dest) >= expected * 0.98:
            print(f"  {local}: already present")
            continue
        tries = 0
        while tries < 5:
            tries += 1
            if download_file(local, bucket_path, expected):
                print(f"  {local}: done")
                break
            print(f"  retrying (attempt {tries + 1}) — download resumes where it stopped")
            time.sleep(2)
        else:
            ok = False
    if not ok:
        print("Some files failed to download. Restart the server to resume.")
    return ok


def run_script(script):
    print(f"- generating {script} ...")
    try:
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, script)],
                           cwd=ROOT, timeout=3600)
        return r.returncode == 0
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {script} failed: {exc}")
        return False


def ensure_dataset(do_download=True):
    """Download raw data + build derived files if anything is missing."""
    if do_download and need_download():
        if not download_dataset():
            print("Continuing with whatever is available "
                  "(some features may not work until all files exist).")
    elif do_download and not need_download():
        print("Dataset: already present.")
    missing_scripts = sorted({s for name, s in DERIVED
                              if not os.path.exists(os.path.join(DATA, name))})
    if missing_scripts:
        print("Derived web files missing; building them "
              "(needs: pip install pandas pyarrow numpy) ...")
        for s in missing_scripts:
            run_script(s)
    if not need_derive():
        print("Derived web files: OK.")


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
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    port = int(args[0]) if args else 8000
    do_download = "--no-download" not in sys.argv
    os.makedirs(DATA, exist_ok=True)

    ensure_dataset(do_download)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Fly Brain Explorer running at http://127.0.0.1:{port}/")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
