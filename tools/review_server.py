"""Loopback-only dataset editor with explicit drafts, approvals, stale-write rejection and history."""
from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ipcam.vision.desk import NAMES

from tools.label_editor import build_editor, digest, import_patch, paths, restore_last_save, stage_correction, review_revision


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()

    def state(self, key):
        image, label = paths(self.root, key)
        review = json.loads((self.root / 'review.json').read_text(encoding='utf-8'))
        entry = review['images'][key]
        return dict(image=key, boxes=[list(map(float, line.split())) for line in
                                     label.read_text(encoding='utf-8').splitlines()],
                    image_sha256=digest(image), label_sha256=digest(label), status=entry['status'],
                    finding=entry.get('finding', ''),
                    review_revision=review_revision(entry),
                    progress=dict(total=len(review['images']),
                                  approved=sum(e.get('status') == 'approved' for e in review['images'].values())))

    def apply(self, action, payload):
        if not isinstance(payload, dict):
            raise ValueError('Review request must be an object')
        if not isinstance(payload.get('review_revision'), str) or not payload['review_revision']:
            raise ValueError('Current review revision required')
        key = payload.get('image', '')
        with self.lock:
            if action == 'approve':
                import_patch(self.root, payload)
            elif action == 'draft':
                if payload.get('names') != list(NAMES):
                    raise ValueError('Class definitions changed')
                if not isinstance(payload.get('reviewer'), str) or not payload['reviewer'].strip():
                    raise ValueError('Reviewer identity required for draft saves')
                if not payload.get('image_sha256') or not payload.get('label_sha256'):
                    raise ValueError('Image and label hashes required')
                stage_correction(self.root, key, payload.get('boxes'), 'Draft saved; exhaustive review pending',
                                 payload['reviewer'], payload['image_sha256'], payload['label_sha256'], draft=True,
                                 expected_revision=payload['review_revision'])
            elif action == 'restore':
                restore_last_save(self.root, key, payload.get('image_sha256'), payload.get('label_sha256'),
                                  payload['review_revision'])
            else:
                raise ValueError('Unknown review action')
            return self.state(key)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root, port=8765):
        self.store = ReviewStore(root)
        self.token = secrets.token_urlsafe(32)
        build_editor(self.store.root)
        super().__init__(('127.0.0.1', port), ReviewHandler)
        self.origin = f'http://127.0.0.1:{self.server_port}'


class ReviewHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        pass

    def reply(self, code, data, content_type='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def local_request(self):
        origin = self.headers.get('Origin')
        return (self.headers.get('Host') == urlsplit(self.server.origin).netloc
                and (origin is None or origin == self.server.origin))

    def do_GET(self):
        if not self.local_request():
            return self.reply(403, dict(error='Local editor origin required'))
        route = unquote(urlsplit(self.path).path)
        root = self.server.store.root
        if route in ('/', '/label-editor.html'):
            # Token lives only in this server-rendered response, never in a URL or dataset file.
            with self.server.store.lock:
                build_editor(root)
                html = (root / 'label-editor.html').read_text(encoding='utf-8')
            html = html.replace('\nconst api=null;\n', '\nconst api=' + json.dumps(dict(token=self.server.token)) + ';\n', 1)
            return self.reply(200, html.encode('utf-8'), 'text/html; charset=utf-8')
        if not route.startswith('/images/'):
            return self.reply(404, dict(error='Not found'))
        try:
            image, _ = paths(root, route.lstrip('/'))
            if image.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                raise ValueError('Unsupported image type')
            self.reply(200, image.read_bytes(), mimetypes.guess_type(image.name)[0] or 'application/octet-stream')
        except (ValueError, OSError):
            self.reply(404, dict(error='Image not found'))

    def do_POST(self):
        if not self.local_request() or not secrets.compare_digest(self.headers.get('X-Review-Token', '').encode('utf-8'),
                                                                 self.server.token.encode('utf-8')):
            return self.reply(403, dict(error='Local editor token required'))
        route = urlsplit(self.path).path
        if route not in ('/api/approve', '/api/draft', '/api/restore'):
            return self.reply(404, dict(error='Unknown action'))
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2*1024*1024:
                return self.reply(413, dict(error='Invalid request size'))
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(415, dict(error='JSON required'))
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Incomplete request')
            payload = json.loads(body)
            result = self.server.store.apply(route.rsplit('/', 1)[1], payload)
            self.reply(200, result)
        except (ValueError, TypeError, KeyError) as exc:
            self.reply(409, dict(error=str(exc)))
        except OSError:
            self.reply(500, dict(error='Save failed; reload the image before retrying'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('dataset_cctv_desk'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    with ReviewServer(args.root, args.port) as server:
        print(f'Label editor: {server.origin}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
