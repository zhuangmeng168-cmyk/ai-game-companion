"""轻量中文检索：字二元组 TF-IDF，无需模型、本地运行。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .config import INDEX_DIR

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM = re.compile(r"[a-z0-9]+")


def terms(text: str) -> list[str]:
    text = (text or "").lower()
    out = _ALNUM.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


class GameIndex:
    def __init__(self, game_id, game_name="", chunks=None, tips=None, built_at=""):
        self.game_id = game_id
        self.game_name = game_name
        self.built_at = built_at
        self.chunks = [dict(c) for c in (chunks or [])]
        self.tips = [dict(t) for t in (tips or [])]
        self._prepare()

    def _prepare(self):
        self._doc_tf = []
        self.df = Counter()
        for chunk in self.chunks:
            tf = Counter(terms(f"{chunk.get('title', '')} {chunk.get('text', '')}"))
            self._doc_tf.append(tf)
            self.df.update(tf.keys())
        self.n = max(1, len(self.chunks))
        self._doc_vecs = [self._vector(tf) for tf in self._doc_tf]
        self._tip_terms = [set(terms(tip.get("text", ""))) for tip in self.tips]

    def _idf(self, term):
        return math.log((1.0 + self.n) / (1.0 + self.df.get(term, 0))) + 1.0

    def _vector(self, tf):
        vec = {term: (1.0 + math.log(count)) * self._idf(term) for term, count in tf.items() if count > 0}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: value / norm for term, value in vec.items()}

    @staticmethod
    def _cosine(a, b):
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def search(self, query, k=3):
        if not query or not self.chunks:
            return []
        qvec = self._vector(Counter(terms(query)))
        scored = []
        for i, vec in enumerate(self._doc_vecs):
            score = self._cosine(qvec, vec)
            if score > 0:
                chunk = dict(self.chunks[i])
                chunk["score"] = round(score, 4)
                scored.append(chunk)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def find_tips(self, query, k=8):
        if not self.tips or not query:
            return []
        qterms = Counter(terms(query))
        if not qterms:
            return []
        weights = {term: self._idf(term) for term in qterms}
        scored = []
        for i, tip_terms in enumerate(self._tip_terms):
            overlap = [t for t in qterms if t in tip_terms]
            if not overlap:
                continue
            score = sum(weights[t] for t in overlap) / (math.sqrt(len(tip_terms) or 1) * math.sqrt(len(qterms)))
            tip = dict(self.tips[i])
            tip["score"] = round(score, 4)
            scored.append(tip)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def to_dict(self):
        return {"game_id": self.game_id, "game_name": self.game_name, "built_at": self.built_at, "chunks": self.chunks, "tips": self.tips}

    @classmethod
    def from_dict(cls, data):
        return cls(game_id=str(data.get("game_id", "")), game_name=str(data.get("game_name", "")), chunks=data.get("chunks") or [], tips=data.get("tips") or [], built_at=str(data.get("built_at", "")))


def load_index(game_id):
    path = INDEX_DIR / f"{game_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return GameIndex.from_dict(data)


def save_index(index):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / f"{index.game_id}.json"
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
