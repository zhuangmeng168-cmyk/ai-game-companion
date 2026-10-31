"""准确率自检：拿测试题真实调用一次当前联网模型，统计建议命中率。

注意：会真实消耗少量 API 费用，由用户在界面上手动触发。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import config as config_module
from . import llm
from .config import GAMES_DIR
from .games import load_games
from .intents import expand
from .persona import few_shot_pairs, get_persona, system_prompt
from .retrieval import load_index
from .style import enforce


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def collect_cases(per_game: int = 2) -> list[dict]:
    cases: list[dict] = []
    for game in load_games():
        path = GAMES_DIR / game["id"] / "testset.jsonl"
        if not path.exists():
            continue
        taken = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or taken >= per_game:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = str(case.get("q", "")).strip()
            if not question:
                continue
            cases.append(
                {
                    "game_id": game["id"],
                    "game_name": game["name"],
                    "q": question,
                    "section": str(case.get("section", "")),
                    "expect": [str(word) for word in case.get("expect", []) if word],
                }
            )
            taken += 1
    return cases


def run_live_eval(per_game: int = 2, max_workers: int = 3) -> dict:
    cfg = config_module.load_config()
    if not str(cfg.get("api_key") or "").strip():
        return {"ok": False, "error": "先配置 API Key 才能做准确率自检"}
    cases = collect_cases(per_game)
    if not cases:
        return {"ok": False, "error": "没有找到测试题"}

    persona = get_persona(cfg.get("persona"))
    max_chars = int(cfg.get("max_chars", 10))
    player_name = str(cfg.get("player_name", "老哥"))

    indexes = {}
    for case in cases:
        game_id = case["game_id"]
        if game_id not in indexes:
            indexes[game_id] = load_index(game_id)

    def one(case: dict) -> dict:
        knowledge: list[str] = []
        index = indexes.get(case["game_id"])
        if index is not None:
            search_text, _intent = expand(case["q"])
            for tip in index.find_tips(search_text, k=3)[:2]:
                knowledge.append(f"{tip.get('section', '')}：{tip.get('text', '')}")
            for chunk in index.search(search_text, k=2)[:1]:
                knowledge.append(f"{chunk.get('title', '')}：{str(chunk.get('text', ''))[:120]}")
        messages = [
            {
                "role": "system",
                "content": system_prompt(persona, case["game_name"], max_chars, player_name, knowledge),
            }
        ]
        messages.extend(few_shot_pairs(persona, max_chars))
        messages.append({"role": "user", "content": case["q"]})
        try:
            raw = llm.chat(cfg, messages)
            reply = enforce(raw, max_chars)
        except llm.LLMError as exc:
            return {**case, "reply": "", "hit": False, "error": str(exc)}
        hit = any(word in reply for word in case["expect"])
        return {**case, "reply": reply, "hit": hit, "error": ""}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, min(6, max_workers))) as pool:
        results = list(pool.map(one, cases))

    valid = [item for item in results if not item["error"]]
    hits = sum(1 for item in valid if item["hit"])
    compliant = sum(1 for item in valid if 0 < len(item["reply"]) <= max_chars)
    errors = len(results) - len(valid)
    return {
        "ok": True,
        "total": len(results),
        "valid": len(valid),
        "errors": errors,
        "hits": hits,
        "accuracy": round(hits / len(valid), 3) if valid else 0,
        "compliance": round(compliant / len(valid), 3) if valid else 0,
        "avg_chars": round(sum(len(item["reply"]) for item in valid) / len(valid), 2) if valid else 0,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "model": str(cfg.get("model", "")),
        "cases": results,
        "note": "真实调用当前模型，会产生少量 API 费用；命中 = 回复包含该题参考关键词。",
        "at": _now_iso(),
    }
