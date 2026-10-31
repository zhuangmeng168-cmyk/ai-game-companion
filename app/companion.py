"""陪玩编排：联网大模型 + 本地攻略检索，保证回答懂游戏、像真人、≤10 字。"""

from __future__ import annotations

import threading
import time

from . import config as config_module
from .games import get_game, load_games
from .intents import expand
from .llm import LLMError, chat
from .persona import few_shot_pairs, get_persona, system_prompt
from .retrieval import GameIndex, load_index
from .style import enforce

SESSION_TTL = 6 * 3600
MAX_SESSIONS = 300
MAX_HISTORY = 8


class Companion:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._indexes: dict[str, GameIndex | None] = {}
        self._sessions: dict[str, dict] = {}
        self._stats = {"replies": 0, "compliant": 0, "llm_ok": 0, "llm_fail": 0}

    def index_for(self, game_id: str) -> GameIndex | None:
        with self._lock:
            if game_id in self._indexes:
                return self._indexes[game_id]
        index = load_index(game_id)
        with self._lock:
            self._indexes[game_id] = index
        return index

    def reload_indexes(self) -> None:
        with self._lock:
            self._indexes.clear()

    def _session(self, session_id: str) -> dict:
        now = time.time()
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                self._gc_sessions(now)
            session = self._sessions.get(session_id)
            if session is None:
                session = {"history": [], "recent": [], "updated": now}
                self._sessions[session_id] = session
            session["updated"] = now
            return session

    def _gc_sessions(self, now: float) -> None:
        stale = [key for key, value in self._sessions.items() if now - value.get("updated", 0) > SESSION_TTL]
        for key in stale:
            self._sessions.pop(key, None)
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = sorted(self._sessions.items(), key=lambda item: item[1].get("updated", 0))
            for key, _ in oldest[: len(self._sessions) - MAX_SESSIONS + 1]:
                self._sessions.pop(key, None)

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def stats(self) -> dict:
        with self._lock:
            data = dict(self._stats)
            data["sessions"] = len(self._sessions)
        return data

    @staticmethod
    def _pick_game(game_id: str | None, games: list[dict], cfg: dict) -> dict | None:
        for candidate in (game_id, cfg.get("last_game"), "wangzhe"):
            game = get_game(candidate, games)
            if game:
                return game
        return games[0] if games else None

    def ask(
        self,
        message: str,
        session_id: str = "default",
        game_id: str | None = None,
        persona_id: str | None = None,
    ) -> dict:
        cfg = config_module.load_config()
        text = (message or "").strip()[:200]
        if not text:
            return {"error": "说点什么吧"}
        if not str(cfg.get("api_key") or "").strip():
            return {
                "error": "还没连接 AI：点右上角 ⚙ 填一个 API Key（DeepSeek 等）",
                "need_setup": True,
            }

        max_chars = int(cfg.get("max_chars", 10))
        games = load_games()
        game = self._pick_game(game_id, games, cfg)
        game_name = game["name"] if game else "游戏"
        persona = get_persona(persona_id or cfg.get("persona"))
        session_id = (session_id or "default")[:64]
        session = self._session(session_id)
        index = self.index_for(game["id"]) if game else None

        search_text, intent = expand(text)
        knowledge_items: list[dict] = []
        knowledge_lines: list[str] = []
        if index is not None:
            for tip in index.find_tips(search_text, k=3)[:2]:
                knowledge_items.append({"text": str(tip.get("text", "")), "section": str(tip.get("section", ""))})
                knowledge_lines.append(f"{tip.get('section', '')}：{tip.get('text', '')}")
            for chunk in index.search(search_text, k=2):
                if len(knowledge_lines) >= 4:
                    break
                snippet = str(chunk.get("text", ""))[:120]
                if snippet:
                    knowledge_lines.append(f"{chunk.get('title', '')}：{snippet}")
                    if len(knowledge_items) < 2:
                        knowledge_items.append(
                            {"text": str(chunk.get("title", "")), "section": str(chunk.get("title", ""))}
                        )

        messages = [
            {
                "role": "system",
                "content": system_prompt(
                    persona,
                    game_name,
                    max_chars,
                    str(cfg.get("player_name", "老哥")),
                    knowledge_lines,
                ),
            }
        ]
        for item in session["history"][-2:]:
            messages.append({"role": "user", "content": item["q"]})
            messages.append({"role": "assistant", "content": item["a"]})
        messages.extend(few_shot_pairs(persona, max_chars))
        messages.append({"role": "user", "content": text})

        started = time.perf_counter()
        try:
            raw = chat(cfg, messages)
        except LLMError as exc:
            with self._lock:
                self._stats["llm_fail"] += 1
            message_text = str(exc)
            return {"error": message_text, "need_setup": "API Key" in message_text}
        if not raw.strip():
            return {"error": "模型没吭声，再问一次？"}

        reply = enforce(raw, max_chars, avoid=session["recent"])
        with self._lock:
            session["history"].append({"q": text, "a": reply})
            session["history"] = session["history"][-MAX_HISTORY:]
            session["recent"] = (session["recent"] + [reply])[-4:]
            self._stats["replies"] += 1
            self._stats["llm_ok"] += 1
            if len(reply) <= max_chars:
                self._stats["compliant"] += 1

        return {
            "reply": reply,
            "chars": len(reply),
            "max_chars": max_chars,
            "compliant": len(reply) <= max_chars,
            "game": {"id": game["id"], "name": game["name"]} if game else {"id": "", "name": "游戏"},
            "persona": {"id": persona["id"], "name": persona["name"], "voice": persona["voice"]},
            "intent": intent,
            "source": "cloud",
            "model": str(cfg.get("model", "")),
            "knowledge": knowledge_items[:2],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "reply_delay_ms": int(cfg.get("reply_delay_ms", 450)),
            "session_id": session_id,
        }
