"""Browser annotation over an SSH tunnel; no Qt, web framework or system packages.

Server: python data/collection/label_web.py PMC9883282 1 --port 8765
Local:  ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
Open the token-bearing localhost URL printed at startup in your local browser.
"""
import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

import numpy as np
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.unet.postprocess import clean_binary, clean_labels

try:
    from .config import DataInfo
except ImportError:
    from config import DataInfo

STATIC = Path(__file__).with_name("web")
EXCLUSION_REASONS = {
    "manual_no_contact": "探头未接触 / 无有效组织回声",
    "manual_non_ultrasound": "非超声画面",
    "manual_unusable_image": "图像损坏或采集失败",
}


class ConflictError(ValueError):
    pass


def cleanup_pmc_annotation(labels):
    """User-requested save cleanup: one target per class, no A/V reassignment.

    Work at native resolution; retain even a one-pixel compressed target. Closing
    uses radius 2, fills enclosed holes, and never invents an absent class.
    """
    ids = np.zeros(labels.shape, dtype=np.uint8)
    ids[labels == 255] = 1
    ids[labels == 128] = 2
    cleaned = clean_labels(ids, min_area=1, closing_radius=2,
                           fill_holes=True, reassign=False)
    # A removed orphan of one class must not be repainted as the other class.
    cleaned[(ids != 0) & (cleaned != ids)] = 0
    out = np.zeros_like(labels)
    for c, value in ((1, 255), (2, 128)):
        out[clean_binary(cleaned == c, min_area=1)] = value
    return out


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".annotation-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class AnnotationStore:
    """Single-server writes; reread CSV on each operation and reject stale tabs."""
    def __init__(self, info):
        self.info = info
        self.root = info.base_dir.resolve()
        self.lock = threading.RLock()
        with self.lock:
            _, rows = self._read()
            if not rows:
                raise ValueError("Annotation CSV is empty; prepare the dataset first")
            if any(r.get("label_mode", "") not in ("", "full3", "vessel", "vein", "artery") for r in rows):
                raise ValueError("Unsupported label_mode in annotation CSV")

    def _read(self):
        with self.info.meta_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        if not fields or not {"filename", "mask_status"} <= set(fields):
            raise ValueError("CSV requires filename and mask_status columns")
        return fields, rows

    def _path(self, value):
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("File is outside the selected dataset")
        return path

    @staticmethod
    def _row(rows, idx):
        if type(idx) is not int or not 0 <= idx < len(rows):
            raise ValueError("Invalid frame index")
        return rows[idx]

    def _image(self, row):
        value = row.get("relative_path") or ""
        # Some older collection CSVs contain paths from another workstation.
        if value and Path(value).exists():
            path = self._path(value)
        else:
            path = self._path(self.info.images_dir / Path(row["filename"]).name)
        with Image.open(path) as im:
            return im.convert("L")

    def _labels(self, row, size):
        value, source = row.get("mask_path"), "reviewed"
        if not value:
            # Model suggestions must never seed held-out annotations.
            value = row.get("suggestion_path") if row.get("split") not in ("val", "test") else None
            source = "suggestion" if value else "blank"
        raw = self._path(value).read_bytes() if value else b""
        if raw:
            with Image.open(io.BytesIO(raw)) as im:
                if im.size != size or im.mode not in ("L", "P"):
                    raise ValueError("Mask dimensions or format do not match the image")
                labels = np.array(im, dtype=np.uint8)
        else:
            labels = np.zeros((size[1], size[0]), dtype=np.uint8)
        if not set(np.unique(labels)) <= {0, *[b["value"] for b in self.brushes(row)]}:
            raise ValueError("Mask has unsupported class values")
        revision = hashlib.sha256(json.dumps(row, sort_keys=True).encode() + raw).hexdigest()
        return labels, source, revision

    @staticmethod
    def brushes(row):
        mode = row.get("label_mode", "")
        if mode == "full3":
            return [dict(name="静脉", value=255, color="#4d9fff"),
                    dict(name="动脉", value=128, color="#ff6068")]
        if mode == "artery":
            return [dict(name="动脉", value=255, color="#ff6068")]
        if mode == "vein":
            return [dict(name="静脉", value=255, color="#4d9fff")]
        # Existing collected phantoms map 255 to untyped vessel, not vein.
        return [dict(name="血管（未分型）", value=255, color="#44dd88")]

    @staticmethod
    def quality(row):
        return dict(excluded=row.get("frame_valid", "").lower() == "false",
                    exclusion_reason=row.get("exclusion_reason", ""),
                    quality_review=row.get("quality_review", ""))

    def queue(self):
        with self.lock:
            _, rows = self._read()
            return dict(dataset=self.info.test_name, frames=[dict(
                id=i, filename=r["filename"], status=r["mask_status"],
                split=r.get("split", ""), subject=r.get("subject_id", ""),
                **self.quality(r)) for i, r in enumerate(rows)])

    def frame(self, idx):
        with self.lock:
            _, rows = self._read()
            row = self._row(rows, idx)
            im = self._image(row)
            labels, source, revision = self._labels(row, im.size)
            png = io.BytesIO()
            im.save(png, format="PNG")
            return dict(id=idx, filename=row["filename"], width=im.width, height=im.height,
                        image=base64.b64encode(png.getvalue()).decode(),
                        labels=base64.b64encode(labels.tobytes()).decode(), source=source,
                        revision=revision, brushes=self.brushes(row), split=row.get("split", ""),
                        status=row["mask_status"], subject=row.get("subject_id", ""),
                        sequence=row.get("sequence_id", ""), source_frame=row.get("frame_index", ""),
                        cleanup_on_save=self.info.test_name == "PMC9883282" and row.get("label_mode") == "full3",
                        **self.quality(row),
                        suggestion_model=Path(row.get("suggestion_checkpoint", "")).stem if source == "suggestion" else "")

    def update(self, payload, skip=False, quality_action=None):
        with self.lock:
            cleanup = None
            fields, rows = self._read()
            row = self._row(rows, payload.get("id"))
            im = self._image(row)
            _, _, revision = self._labels(row, im.size)
            if payload.get("revision") != revision:
                raise ConflictError("此帧已被其他窗口修改，请重新载入后再保存。")
            if quality_action is not None:
                if quality_action not in ("exclude", "restore"):
                    raise ValueError("Unknown quality action")
                if quality_action == "exclude":
                    reason = payload.get("reason", "manual_no_contact")
                    if reason not in EXCLUSION_REASONS:
                        raise ValueError("请选择无效帧原因；血管消失或压扁应保留为有效帧。")
                    if self.quality(row)["excluded"]:
                        raise ValueError("此帧已经排除，可在已排除队列恢复。")
                    row.update(frame_valid="false", exclusion_reason=reason,
                               quality_review="invalid", mask_status="pass")
                else:
                    if not self.quality(row)["excluded"]:
                        raise ValueError("此帧没有被排除。")
                    # Retain the old mask, but require explicit annotation confirmation.
                    row.update(frame_valid="true", exclusion_reason="",
                               quality_review="valid", mask_status="ND")
                for field in ("frame_valid", "exclusion_reason", "quality_review"):
                    if field not in fields:
                        fields.append(field)
            elif self.quality(row)["excluded"]:
                raise ValueError("此帧已排除。请先恢复为有效帧，核对标注后再保存。")
            elif skip:
                row["mask_status"] = "pass"
            else:
                try:
                    raw = base64.b64decode(payload.get("labels", ""), validate=True)
                except (ValueError, TypeError, binascii.Error) as e:
                    raise ValueError("Invalid mask encoding") from e
                if len(raw) != im.width * im.height:
                    raise ValueError("Mask dimensions do not match the source image")
                labels = np.frombuffer(raw, dtype=np.uint8).reshape(im.height, im.width)
                if not set(np.unique(labels)) <= {0, *[b["value"] for b in self.brushes(row)]}:
                    raise ValueError("Mask contains unsupported class values")
                if self.info.test_name == "PMC9883282" and row.get("label_mode") == "full3":
                    cleaned = cleanup_pmc_annotation(labels)
                    cleanup = {"changed_pixels": int(np.count_nonzero(cleaned != labels)),
                               "closing_radius": 2, "min_area": 1, "max_components_per_class": 1}
                    labels = cleaned
                path = self._path(self.info.masks_dir / (Path(row["filename"]).stem + "_mask.png"))
                png = io.BytesIO()
                Image.fromarray(labels).save(png, format="PNG")
                # Write mask first: failure cannot mark an unwritten mask as reviewed.
                atomic_write(path, png.getvalue())
                if "mask_path" not in fields:
                    fields.append("mask_path")
                row["mask_path"] = str(path)
                row["mask_status"] = "true"
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            atomic_write(self.info.meta_file, stream.getvalue().encode("utf-8"))
            return {"ok": True, "status": row["mask_status"],
                    "cleanup": cleanup,
                    "revision": self._labels(row, im.size)[2], **self.quality(row)}


def make_server(store, port=8765, token=None, stores=None, discover_names=()):
    token = token or secrets.token_urlsafe(24)
    registry = dict(stores or {})
    registry[store.info.test_name] = store
    registry_lock = threading.RLock()

    def datasets():
        with registry_lock:
            for name in discover_names:
                if name not in registry:
                    info = DataInfo(name, '1')
                    if info.meta_file.exists():
                        registry[name] = AnnotationStore(info)
            return dict(registry)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # URLs/headers can contain credentials; do not log them.

        def reply(self, status, body, mime="application/json; charset=utf-8"):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def dispatch(self, post=False):
            try:
                url = urlsplit(self.path)
                assets = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if not post and url.path in assets:
                    name, mime = assets[url.path]
                    return self.reply(200, (STATIC / name).read_bytes(), mime)
                if not secrets.compare_digest(self.headers.get("X-Annotation-Token", ""), token):
                    return self.reply(403, {"error": "连接令牌无效，请打开服务器启动时输出的完整链接。"})
                available = datasets()
                if not post and url.path == '/api/datasets':
                    return self.reply(200, {'default': store.info.test_name, 'datasets': list(available)})
                query = parse_qs(url.query)
                name = query.get('dataset', [store.info.test_name])[0]
                if name not in available:
                    raise ValueError('Unknown or not yet prepared dataset')
                selected_store = available[name]
                if not post and url.path == "/api/queue":
                    return self.reply(200, selected_store.queue())
                if not post and url.path == "/api/frame":
                    idx = int(parse_qs(url.query).get("id", [""])[0])
                    return self.reply(200, selected_store.frame(idx))
                if post and url.path in ("/api/save", "/api/skip", "/api/exclude", "/api/restore"):
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 16 * 1024 * 1024:
                        return self.reply(413, {"error": "Invalid request size"})
                    self.connection.settimeout(30)
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object")
                    action = url.path.rsplit("/", 1)[1]
                    return self.reply(200, selected_store.update(payload, skip=action == "skip",
                                      quality_action=action if action in ("exclude", "restore") else None))
                self.reply(404, {"error": "Not found"})
            except ConflictError as e:
                self.reply(409, {"error": str(e)})
            except (ValueError, KeyError, OSError) as e:
                self.reply(400, {"error": str(e)})

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(post=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.token = token
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_name", nargs="?", default="phantom_taobao")
    parser.add_argument("test_id", nargs="?", default="1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument('--all-datasets', action='store_true', help='Discover prepared public video queues on this port')
    args = parser.parse_args()
    if any(Path(v).name != v or v in (".", "..") for v in (args.test_name, args.test_id)):
        parser.error("Dataset name and test ID must be simple names")
    try:
        store = AnnotationStore(DataInfo(args.test_name, args.test_id))
        server = make_server(store, args.port, discover_names=('PMC9883282', 'ThrombUS', 'Regional-US', 'TUS-REC2024') if args.all_datasets else ())
    except (ValueError, OSError) as e:
        parser.error(str(e))
    port = server.server_address[1]
    print(f"Annotation server: {args.test_name}; listening only on 127.0.0.1:{port}", flush=True)
    print(f"Local SSH tunnel: ssh -N -L {port}:127.0.0.1:{port} USER@SERVER", flush=True)
    print(f"Open locally: http://127.0.0.1:{port}/#token={server.token}", flush=True)
    print("Keep this process running. Ctrl+C stops it. Do not edit this CSV with the desktop tool simultaneously.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
