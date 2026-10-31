"""本地 HTTP 服务：静态页面 + JSON API，默认只监听 127.0.0.1。"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import __app_name__, __version__
from . import benchmark
from . import config as config_module
from . import training
from .companion import Companion
from .games import get_game, load_games, public_game
from .llm import LLMError, chat
from .persona import public_personas
from .screen_watch import ScreenWatcher

COMPANION = Companion()
SCREEN = ScreenWatcher()
STARTED_AT = time.time()
CONFIG_KEYS = {
    "base_url", "api_key", "model", "temperature", "max_chars", "persona",
    "player_name", "reply_delay_ms", "voice", "speech_voice", "wake_words",
    "vision_base_url", "vision_api_key", "vision_model",
    "screen_interval", "screen_speak", "screen_skip_static",
}


class DaziServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _default_game(games, cfg):
    for candidate in (cfg.get("last_game"), "wangzhe"):
        if candidate and get_game(candidate, games):
            return candidate
    return games[0]["id"] if games else ""


def _bootstrap():
    cfg = config_module.load_config()
    games = load_games()
    return {
        "app": {"name": __app_name__, "version": __version__, "started_at": STARTED_AT},
        "config": config_module.public_config(cfg),
        "personas": public_personas(),
        "games": [public_game(game) for game in games],
        "default_game": _default_game(games, cfg),
        "training": training.status(),
        "stats": COMPANION.stats(),
        "screen": {"available": sys.platform == "win32", "stats": SCREEN.stats()},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"Dazi/{__version__}"

    def log_message(self, fmt, *args):
        if config_module.load_config().get("debug"):
            super().log_message(fmt, *args)

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        if length > 30_000_000:
            raise ValueError("请求太大（文档上限约 15MB）")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求不是合法 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("请求格式不对")
        return data

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._handle_api("GET", path)
        else:
            self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            self._send_json(404, {"error": "not found"})
            return
        self._handle_api("POST", path)

    def _handle_api(self, method, path):
        try:
            payload = self._api_get(path) if method == "GET" else self._api_post(path)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception:
            traceback.print_exc()
            self._send_json(500, {"error": "服务出错了，看一下控制台日志"})
            return
        if payload is None:
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, payload)

    def _api_get(self, path):
        if path == "/api/health":
            return {"ok": True, "name": __app_name__, "version": __version__, "uptime": int(time.time() - STARTED_AT)}
        if path == "/api/bootstrap":
            return _bootstrap()
        if path == "/api/config":
            return {"config": config_module.public_config()}
        if path == "/api/train/status":
            return training.status()
        if path == "/api/stats":
            return {"stats": COMPANION.stats(), "screen": SCREEN.stats()}
        return None

    def _api_post(self, path):
        body = self._read_json()
        if path == "/api/chat":
            message = str(body.get("message") or "").strip()
            if not message:
                raise ValueError("消息不能为空")
            session_id = str(body.get("session_id") or "default")[:64]
            game_id = body.get("game_id") or None
            persona_id = body.get("persona") or None
            result = COMPANION.ask(message, session_id=session_id, game_id=game_id, persona_id=persona_id)
            if game_id and result.get("reply") and get_game(game_id, load_games()):
                config_module.save_config({"last_game": game_id})
            return result
        if path == "/api/model/test":
            cfg = config_module.load_config()
            patch = {}
            for key in ("base_url", "model"):
                value = str(body.get(key) or "").strip()
                if value:
                    patch[key] = value
            api_key = str(body.get("api_key") or "").strip()
            if api_key:
                patch["api_key"] = api_key
            cfg.update(patch)
            if not str(cfg.get("api_key") or "").strip():
                return {"ok": False, "error": "还没填 API Key"}
            started = time.perf_counter()
            try:
                reply = chat(cfg, [{"role": "system", "content": "你是连接测试助手，只回复两个字：正常"}, {"role": "user", "content": "测试"}], timeout=30.0)
            except LLMError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "reply": str(reply or "")[:30], "latency_ms": int((time.perf_counter() - started) * 1000), "model": cfg.get("model", ""), "base_url": cfg.get("base_url", "")}
        if path == "/api/config":
            patch = {key: body[key] for key in CONFIG_KEYS if key in body}
            for secret in ("api_key", "vision_api_key"):
                if secret in patch:
                    value = str(patch.get(secret) or "").strip()
                    if value:
                        patch[secret] = value
                    else:
                        patch.pop(secret)
            cfg = config_module.save_config(patch)
            return {"ok": True, "config": config_module.public_config(cfg)}
        if path == "/api/vision/analyze":
            return SCREEN.analyze(body.get("game_id") or None, body.get("persona") or None)
        if path == "/api/vision/reset":
            SCREEN.reset()
            return {"ok": True}
        if path == "/api/eval/live":
            per_game = max(1, min(4, int(body.get("per_game") or 2)))
            return benchmark.run_live_eval(per_game=per_game)
        if path == "/api/train":
            action = str(body.get("action") or "").strip()
            game_id = body.get("game_id") or None
            if action == "ingest":
                result = training.ingest(game_id)
                COMPANION.reload_indexes()
            elif action == "sft":
                result = training.build_sft()
            elif action == "eval":
                result = training.evaluate()
            else:
                raise ValueError("未知训练动作")
            return result
        if path == "/api/games/create":
            game = training.create_game(name=str(body.get("name") or ""), genre=str(body.get("genre") or ""), aliases=str(body.get("aliases") or ""), knowledge=str(body.get("knowledge") or ""))
            COMPANION.reload_indexes()
            return {"ok": True, "game": game}
        if path == "/api/games/import-document":
            raw = str(body.get("data") or "")
            binary = None
            if raw:
                try:
                    binary = base64.b64decode(raw, validate=True)
                except (ValueError, TypeError) as exc:
                    raise ValueError("文档数据无效，请重新选择文件") from exc
            result = training.import_document(name=str(body.get("name") or ""), genre=str(body.get("genre") or ""), aliases=str(body.get("aliases") or ""), filename=str(body.get("filename") or ""), data=binary, text=str(body.get("text") or ""), target_game_id=body.get("target_game_id") or None)
            COMPANION.reload_indexes()
            return result
        if path == "/api/reset":
            COMPANION.reset_session(str(body.get("session_id") or "default")[:64])
            return {"ok": True}
        return None

    def _serve_static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        if path == "/favicon.ico":
            icon = config_module.BUNDLE_DIR / "assets" / "dazi.ico"
            if icon.is_file():
                data = icon.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/x-icon")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(204)
                self.end_headers()
            return
        web_root = config_module.WEB_DIR.resolve()
        target = (web_root / path.lstrip("/")).resolve()
        try:
            inside = target == web_root or target.is_relative_to(web_root)
        except AttributeError:
            inside = str(target).startswith(str(web_root))
        if not inside:
            self._send_json(403, {"error": "forbidden"})
            return
        if not target.is_file():
            self._send_json(404, {"error": "not found"})
            return
        try:
            data = target.read_bytes()
        except OSError:
            self._send_json(500, {"error": "读取文件失败"})
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def create_server(host="127.0.0.1", port=8765):
    return DaziServer((host, port), Handler)


def serve(host="127.0.0.1", port=8765, open_browser=False):
    server = create_server(host, port)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    print("")
    print(f"  {__app_name__} · AI 语音陪玩  v{__version__}")
    print(f"  已启动：{url}")
    print("  先在右上角设置里填 API Key（DeepSeek 等），再开玩")
    print("  停止服务：Ctrl+C")
    print("")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止，再见～")
    finally:
        server.server_close()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="python -m app.server", description="搭子 · AI 语音陪玩")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
