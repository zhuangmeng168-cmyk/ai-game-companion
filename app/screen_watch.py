"""定时看屏：截图 → 画面变化判断 → 视觉模型建议。"""

from __future__ import annotations

import threading

from . import config as config_module
from . import screen, vision
from .games import get_game, load_games
from .persona import get_persona

STATIC_THRESHOLD = 6.0


class ScreenWatcher:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_signature: bytes | None = None
        self._last_advice = ""
        self._stats = {"calls": 0, "skipped": 0, "errors": 0}

    def reset(self) -> None:
        with self._lock:
            self._last_signature = None
            self._last_advice = ""

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def analyze(self, game_id: str | None = None, persona_id: str | None = None) -> dict:
        cfg = config_module.load_config()
        if not str(cfg.get("vision_model") or "").strip():
            return {"error": "还没配置视觉模型：点右上角 ⚙ 填一个支持看图的模型", "need_setup": True}
        if not (str(cfg.get("vision_api_key") or "").strip() or str(cfg.get("api_key") or "").strip()):
            return {"error": "视觉模型没有 API Key，去设置里填一下", "need_setup": True}
        try:
            shot = screen.capture()
        except screen.ScreenError as exc:
            with self._lock:
                self._stats["errors"] += 1
            return {"error": f"截屏失败：{exc}", "need_setup": False}
        max_chars = int(cfg.get("max_chars", 10))
        with self._lock:
            previous = self._last_signature
            if cfg.get("screen_skip_static", True) and previous is not None:
                if screen.distance(shot["signature"], previous) < STATIC_THRESHOLD:
                    self._last_signature = shot["signature"]
                    self._stats["skipped"] += 1
                    return {"skipped": True, "reason": "画面没明显变化", "elapsed_ms": shot["elapsed_ms"]}
            self._last_signature = shot["signature"]
            recent = self._last_advice
        games = load_games()
        game = get_game(game_id, games) or get_game(cfg.get("last_game"), games) or get_game("wangzhe", games) or (games[0] if games else None)
        game_name = game["name"] if game else "游戏"
        persona = get_persona(persona_id or cfg.get("persona"))
        try:
            result = vision.analyze(cfg, shot["png"], game_name, persona_id or cfg.get("persona"), max_chars, recent)
        except vision.VisionError as exc:
            with self._lock:
                self._stats["errors"] += 1
            message = str(exc)
            return {"error": message, "need_setup": ("视觉模型" in message or "API Key" in message)}
        with self._lock:
            self._stats["calls"] += 1
            self._last_advice = result["advice"]
        return {
            "advice": result["advice"], "scene": result["scene"], "chars": len(result["advice"]),
            "compliant": len(result["advice"]) <= max_chars, "max_chars": max_chars,
            "model": result["model"], "latency_ms": result["latency_ms"],
            "capture": {"width": shot["width"], "height": shot["height"], "elapsed_ms": shot["elapsed_ms"]},
            "game": {"id": game["id"], "name": game["name"]} if game else {"id": "", "name": "游戏"},
            "persona": {"id": persona["id"], "name": persona["name"], "voice": persona["voice"]},
            "source": "vision", "reply_delay_ms": 0,
        }
