"""把任意模型的输出整理成真人陪玩短句：默认硬上限 10 个字。"""

from __future__ import annotations

import re

DEFAULT_MAX_CHARS = 10

_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]+")
_ROLE_PREFIX = re.compile(r"^\s*(?:搭子|小暖|老白|阿元|AI|助手|系统)\s*[:：]\s*")
_QUOTES = re.compile(r"^[\s\"'“”‘’「」『』【】]+|[\s\"'“”‘’「」『』【】]+$")
_SPACES = re.compile(r"[ \t\u3000]+")
_CLAUSE_SPLIT = re.compile(r"[，。！？；,.!?;、\n]+")
_MULTI_PUNCT = re.compile(r"[，。！？；,.!?;、]{2,}")
_EDGE_PUNCT = "，。！？；,.!?;、 "


def clean(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = _CODE_FENCE.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _ROLE_PREFIX.sub("", text)
    text = text.replace("*", "").replace("#", "").replace(">", "")
    text = _EMOJI.sub("", text)
    text = _SPACES.sub(" ", text).strip()
    text = _QUOTES.sub("", text).strip()
    text = _MULTI_PUNCT.sub(lambda m: m.group(0)[0], text)
    return text


def count(text: str) -> int:
    return len(clean(text))


def compliant(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> bool:
    cleaned = clean(text)
    return bool(cleaned) and len(cleaned) <= int(max_chars)


def split_clauses(text: str) -> list[str]:
    return [part.strip() for part in _CLAUSE_SPLIT.split(clean(text)) if part.strip()]


def enforce(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    avoid: tuple | list | set = (),
    fallback=None,
) -> str:
    """尽最大努力把回复压进 max_chars：整句可用就用整句，否则取一个短分句。"""
    max_chars = int(max_chars)
    avoid_set = {clean(item) for item in (avoid or ()) if item}
    cleaned = clean(text)

    if cleaned and len(cleaned) <= max_chars and cleaned not in avoid_set:
        return cleaned

    candidates = [
        clause
        for clause in split_clauses(cleaned)
        if 2 <= len(clause) <= max_chars and clause not in avoid_set
    ]
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    if fallback is not None:
        try:
            fb = clean(fallback())
        except Exception:
            fb = ""
        if fb and len(fb) <= max_chars and fb not in avoid_set:
            return fb

    if cleaned:
        clipped = cleaned[:max_chars].strip(_EDGE_PUNCT)
        if clipped:
            return clipped
    return "嗯嗯"
