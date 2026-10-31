"""训练流水线：攻略入库 → SFT 训练集 → 评估。

用法：
    python -m app.training ingest      # 解析所有 knowledge.md，建立检索索引
    python -m app.training sft         # 生成 ≤10 字风格的 SFT 数据集
    python -m app.training eval        # 离线评估攻略检索 / 训练集合规
    python -m app.training all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import GAMES_DIR, SFT_DIR
from .docx_import import extract_document
from .games import get_game, load_games
from .intents import expand
from .persona import get_persona, system_prompt
from .retrieval import GameIndex, load_index, save_index
from .style import enforce

_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SLUG = re.compile(r"[^a-z0-9]+")

QUESTION_BANK: dict[str, list[str]] = {
    "对线": ["对线怎么打？", "对线期注意什么？"],
    "团战": ["团战怎么打？", "开团怎么找机会？"],
    "站位": ["团战站哪？", "输出位怎么站？"],
    "出装": ["出什么装备？", "装备怎么选？"],
    "装备": ["出什么装备？", "装备怎么选？"],
    "打野": ["打野怎么带节奏？", "打野怎么抓人？"],
    "游走": ["打野怎么抓人？", "怎么游走？"],
    "资源": ["资源怎么控？", "什么时候开龙？"],
    "小龙": ["小龙要不要接？"],
    "大龙": ["什么时候打大龙？"],
    "运营": ["什么时候打大龙？", "兵线怎么处理？"],
    "心态": ["心态崩了怎么办？", "逆风怎么翻？"],
    "开局": ["开局要干嘛？", "前期怎么打？"],
    "英雄": ["选什么英雄？", "阵容怎么搭？"],
    "阵容": ["阵容怎么搭？", "选什么英雄？"],
    "配队": ["配队怎么配？", "队伍怎么搭？"],
    "养成": ["先练谁？", "怎么养成？"],
    "圣遗物": ["圣遗物怎么刷？"],
    "深渊": ["深渊怎么打？"],
    "战斗": ["战斗怎么打输出？", "打boss注意什么？"],
    "元素": ["元素反应怎么打？", "战斗怎么打输出？"],
    "跳伞": ["跳哪比较好？", "落点怎么选？"],
    "落点": ["跳哪比较好？"],
    "搜物资": ["搜什么物资？"],
    "转移": ["怎么进圈？", "跑毒注意什么？"],
    "圈": ["怎么进圈？", "跑毒注意什么？"],
    "决赛圈": ["决赛圈怎么打？"],
    "队友": ["怎么和队友配合？"],
    "交战": ["刚枪打不过怎么办？", "怎么刚枪？"],
    "探索": ["体力怎么花？", "探索注意什么？"],
    "日常": ["日常怎么安排？", "树脂怎么用？"],
    "补刀": ["怎么补刀？"],
}

DEFAULT_QUESTIONS = ["这波怎么打？", "给个建议", "现在该干嘛？"]

STYLE_SAMPLES: list[tuple[str, str]] = [
    ("这波怎么打？", "别急，等他交技能"),
    ("我崩了怎么办？", "稳住，还能翻"),
    ("出什么装？", "先出核心装"),
    ("对面打野在哪？", "小心野区"),
    ("谢谢你", "客气啥"),
    ("在吗？", "在呢"),
    ("我好累", "歇会儿吧"),
    ("输了", "没事，再来"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_markdown(text, game_name):
    sections = []
    current = None
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        heading = _HEADING.match(line)
        if heading:
            if current is not None:
                sections.append(current)
            current = {"title": heading.group(2).strip() or game_name, "lines": [], "tips": []}
            continue
        if not line.strip():
            continue
        if current is None:
            current = {"title": game_name, "lines": [], "tips": []}
        bullet = _BULLET.match(line)
        body = bullet.group(1).strip() if bullet else line.strip()
        current["lines"].append(body)
        if bullet and 2 <= len(body) <= 10:
            current["tips"].append(body)
    if current is not None:
        sections.append(current)
    return [s for s in sections if s["lines"]]


def build_index(game):
    path = GAMES_DIR / game["id"] / "knowledge.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    sections = parse_markdown(text, game.get("name", game["id"]))
    chunks = []
    tips = []
    for number, section in enumerate(sections):
        body = "。".join(section["lines"])
        section_tips = list(section["tips"])
        for tip_text in section_tips:
            tips.append({"text": tip_text, "section": section["title"]})
        chunks.append({
            "id": f"{game['id']}-{number}",
            "title": section["title"],
            "text": f"{section['title']}。{body}",
            "tips": section_tips,
        })
    return GameIndex(game_id=game["id"], game_name=game.get("name", ""), chunks=chunks, tips=tips, built_at=now_iso())


def ingest(game_id=None):
    games = load_games()
    if game_id:
        games = [g for g in games if g["id"] == game_id]
        if not games:
            return {"ok": False, "error": f"没找到游戏 {game_id}"}
    results = []
    for game in games:
        index = build_index(game)
        save_index(index)
        results.append({"id": game["id"], "name": game["name"], "chunks": len(index.chunks), "tips": len(index.tips)})
    return {"ok": True, "action": "ingest", "games": results, "total_chunks": sum(r["chunks"] for r in results), "total_tips": sum(r["tips"] for r in results), "at": now_iso()}


def _questions_for(title, game_name):
    for keyword, questions in QUESTION_BANK.items():
        if keyword in title:
            return questions
    return [f"{game_name}怎么玩？", "有什么技巧？"]


def build_sft():
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    games = load_games()
    samples = []
    seen = set()
    persona_ids = ["wenrou", "dulao", "yuanqi"]

    def add(persona_id, game_name, question, answer, source):
        persona = get_persona(persona_id)
        answer = enforce(answer, 10)
        if not answer or len(answer) > 10:
            return
        system = system_prompt(persona, game_name, 10, "老哥", None)
        key = (system, question, answer)
        if key in seen:
            return
        seen.add(key)
        samples.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}], "meta": {"game": game_name, "persona": persona_id, "source": source}})

    for game in games:
        index = build_index(game)
        for chunk in index.chunks:
            tips = chunk.get("tips") or []
            if not tips:
                continue
            questions = _questions_for(chunk["title"], game["name"])
            for number, tip_text in enumerate(tips[:2]):
                question = questions[number % len(questions)]
                persona_id = persona_ids[(number + len(samples)) % len(persona_ids)]
                add(persona_id, game["name"], question, tip_text, "knowledge")

    for persona_id in persona_ids:
        for question, answer in STYLE_SAMPLES:
            add(persona_id, "游戏", question, answer, "style")

    path = SFT_DIR / "train.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    by_persona = {}
    by_game = {}
    for sample in samples:
        by_persona[sample["meta"]["persona"]] = by_persona.get(sample["meta"]["persona"], 0) + 1
        by_game[sample["meta"]["game"]] = by_game.get(sample["meta"]["game"], 0) + 1
    stats = {"samples": len(samples), "by_persona": by_persona, "by_game": by_game, "max_chars": 10, "generated_at": now_iso()}
    (SFT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "action": "sft", **stats}


def evaluate(limit_examples=8):
    games = load_games()
    total = 0
    retrieval_hit = 0
    examples = []
    per_game = {}
    for game in games:
        testset_path = GAMES_DIR / game["id"] / "testset.jsonl"
        if not testset_path.exists():
            continue
        index = load_index(game["id"]) or build_index(game)
        game_total = game_hit = 0
        for line in testset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            query = str(case.get("q", "")).strip()
            if not query:
                continue
            search_text, _intent = expand(query)
            chunks = index.search(search_text, k=2)
            expected_section = str(case.get("section", ""))
            hit = any(expected_section and expected_section in str(chunk.get("title", "")) for chunk in chunks)
            total += 1
            game_total += 1
            retrieval_hit += int(hit)
            game_hit += int(hit)
            if len(examples) < limit_examples:
                examples.append({"game": game["name"], "q": query, "section": expected_section, "retrieval_hit": hit, "top_sections": [str(chunk.get("title", "")) for chunk in chunks]})
        per_game[game["id"]] = {"name": game["name"], "total": game_total, "retrieval_hit_rate": round(game_hit / game_total, 3) if game_total else 0}
    sft_samples = 0
    sft_style_ok = 0
    sft_path = SFT_DIR / "train.jsonl"
    if sft_path.exists():
        for line in sft_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            sft_samples += 1
            messages = sample.get("messages") or []
            content = str(messages[-1].get("content", "")) if messages else ""
            if 0 < len(content) <= 10:
                sft_style_ok += 1
    report = {"ok": True, "action": "eval", "total": total, "retrieval_hit_rate": round(retrieval_hit / total, 3) if total else 0, "sft_samples": sft_samples, "sft_style_compliance": round(sft_style_ok / sft_samples, 3) if sft_samples else 0, "by_game": per_game, "examples": examples, "note": "离线评估只衡量攻略检索命中与训练集短句合规，不调用线上模型。", "at": now_iso()}
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    (SFT_DIR / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _slug(name):
    slug = _SLUG.sub("-", name.lower()).strip("-")
    if len(slug) >= 2:
        return slug[:24]
    return "game-" + hashlib.md5(name.encode("utf-8")).hexdigest()[:8]


def normalize_import(text):
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or _BULLET.match(line):
            lines.append(stripped)
        elif len(stripped) <= 10:
            lines.append(f"- {stripped}")
        else:
            lines.append(stripped)
    return "\n".join(lines)


def create_game(name, genre="", aliases="", knowledge=""):
    name = str(name or "").strip()[:30]
    if not name:
        raise ValueError("游戏名不能为空")
    game_id = _slug(name)
    target = GAMES_DIR / game_id
    suffix = 2
    while target.exists():
        target = GAMES_DIR / f"{game_id}-{suffix}"
        suffix += 1
    game_id = target.name
    target.mkdir(parents=True, exist_ok=True)
    alias_list = [item.strip() for item in re.split(r"[,，/|、\s]+", str(aliases or "")) if item.strip()]
    profile = {"id": game_id, "name": name, "genre": genre or "自定义", "aliases": alias_list or [name], "window_keywords": alias_list or [name], "intro": f"{name} · 玩家导入的陪玩知识", "terms": [], "quick_questions": ["这波怎么打？", "有什么技巧？", "我现在该干嘛？"], "voice": {"rate": 1.0, "pitch": 1.05}}
    (target / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    body = normalize_import(knowledge)
    markdown = f"# {name} · 陪玩知识\n\n## 玩家导入攻略\n{body}\n" if body else f"# {name} · 陪玩知识\n\n## 玩家导入攻略\n- 这游戏先稳住\n"
    (target / "knowledge.md").write_text(markdown, encoding="utf-8")
    ingest(game_id)
    return profile


def import_document(name="", genre="", aliases="", filename="", data=None, text="", target_game_id=None):
    raw_text = str(text or "").strip()
    if not raw_text:
        if data is None:
            raise ValueError("没有收到文档内容")
        if len(data) > 15_000_000:
            raise ValueError("文档太大，上限 15MB")
        raw_text = extract_document(filename, data).strip()
    if not raw_text:
        raise ValueError("文档里没有可用的文字")
    if len(raw_text) > 2_000_000:
        raise ValueError("解析后的攻略太长（超过 200 万字符），请拆成多个文档")
    if target_game_id:
        game = get_game(target_game_id)
        if not game:
            raise ValueError("要追加攻略的游戏不存在")
        if not re.search(r"(?m)^(?:#{1,6}\s|[-*•]\s)", raw_text):
            raw_text = normalize_import(raw_text)
        knowledge_path = GAMES_DIR / target_game_id / "knowledge.md"
        stamp = now_iso()[:10]
        with knowledge_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n## 文档导入 {stamp}\n{raw_text}\n")
        ingested = ingest(target_game_id)
        return {"ok": True, "mode": "append", "game": game, "chars": len(raw_text), "ingest": ingested}
    game_name = (str(name or "").strip() or Path(filename).stem or "导入攻略")[:30]
    markdown = f"# {game_name} · 陪玩知识\n\n## 导入文档\n{raw_text}\n"
    game = create_game(name=game_name, genre=genre, aliases=aliases, knowledge=markdown)
    return {"ok": True, "mode": "new", "game": game, "chars": len(raw_text)}


def status():
    games = load_games()
    game_rows = []
    for game in games:
        index = load_index(game["id"])
        game_rows.append({"id": game["id"], "name": game["name"], "genre": game.get("genre", ""), "chunks": len(index.chunks) if index else 0, "tips": len(index.tips) if index else 0, "built_at": index.built_at if index else ""})
    sft_stats = {}
    stats_path = SFT_DIR / "stats.json"
    if stats_path.exists():
        try:
            sft_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sft_stats = {}
    eval_report = {}
    eval_path = SFT_DIR / "eval_report.json"
    if eval_path.exists():
        try:
            eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            eval_report = {}
    return {"games": game_rows, "sft": sft_stats, "eval": eval_report}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="python -m app.training", description="搭子训练流水线")
    parser.add_argument("action", choices=["ingest", "sft", "eval", "all"])
    parser.add_argument("--game", default=None, help="只处理某个游戏 id")
    args = parser.parse_args(argv)
    if args.action == "ingest":
        result = ingest(args.game)
    elif args.action == "sft":
        result = build_sft()
    elif args.action == "eval":
        result = evaluate()
    else:
        result = {"ingest": ingest(args.game), "sft": build_sft(), "eval": evaluate()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
