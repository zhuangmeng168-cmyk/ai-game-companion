"""配置读写：开发版写 data/，安装版写 %LOCALAPPDATA%\Dazi\data，API Key 只存在本机。"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _bundle_dir() -> Path:
    """PyInstaller 打包后资源在 _MEIPASS；开发时在项目根目录。"""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def _user_data_dir() -> Path:
    override = os.environ.get("DAZI_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if _is_frozen():
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "Dazi" / "data"
    return _bundle_dir() / "data"


BUNDLE_DIR = _bundle_dir()
DATA_DIR = _user_data_dir()
GAMES_DIR = DATA_DIR / "games"
INDEX_DIR = DATA_DIR / "index"
SFT_DIR = DATA_DIR / "sft"
WEB_DIR = BUNDLE_DIR / "web"
CONFIG_PATH = DATA_DIR / "config.json"


def ensure_user_data() -> Path:
    """安装版第一次运行时，把自带攻略和默认配置复制到可写目录。"""
    if (DATA_DIR / "games").exists() and CONFIG_PATH.exists():
        return DATA_DIR
    source = BUNDLE_DIR / "data"
    if source.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, DATA_DIR, dirs_exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


if _is_frozen():
    ensure_user_data()

DEFAULT_CONFIG = {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "",
    "model": "deepseek-chat",
    "temperature": 0.9,
    "max_chars": 10,
    "persona": "wenrou",
    "player_name": "老哥",
    "reply_delay_ms": 450,
    "voice": {"enabled": True, "rate": 1.0, "pitch": 1.0},
    "speech_voice": "",
    "last_game": "wangzhe",
    "wake_words": ["搭子"],
    "vision_base_url": "",
    "vision_api_key": "",
    "vision_model": "",
    "screen_interval": 5,
    "screen_speak": True,
    "screen_skip_static": True,
}

_lock = threading.RLock()
_WAKE_SPLIT = re.compile(r"[,，、/、\s]+")


def normalize_wake_words(value) -> list[str]:
    """唤醒词规整：支持字符串或列表，去重、限长、最多 8 个，空则回默认「搭子」。"""
    if isinstance(value, str):
        parts = _WAKE_SPLIT.split(value)
    elif isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            parts.extend(_WAKE_SPLIT.split(str(item)))
    else:
        parts = []
    cleaned: list[str] = []
    for part in parts:
        word = part.strip()[:8]
        if word and word not in cleaned:
            cleaned.append(word)
    return (cleaned or ["搭子"])[:8]


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def read_raw() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def load_config() -> dict:
    with _lock:
        return _merge(DEFAULT_CONFIG, read_raw())


def save_config(patch: dict) -> dict:
    with _lock:
        cfg = _merge(load_config(), patch or {})
        try:
            cfg["max_chars"] = max(4, min(30, int(cfg.get("max_chars", 10))))
        except (TypeError, ValueError):
            cfg["max_chars"] = 10
        try:
            cfg["temperature"] = max(0.0, min(2.0, float(cfg.get("temperature", 0.9))))
        except (TypeError, ValueError):
            cfg["temperature"] = 0.9
        try:
            cfg["reply_delay_ms"] = max(0, min(3000, int(cfg.get("reply_delay_ms", 450))))
        except (TypeError, ValueError):
            cfg["reply_delay_ms"] = 450
        cfg["base_url"] = str(cfg.get("base_url", "")).strip()
        cfg["model"] = str(cfg.get("model", "")).strip()
        cfg["speech_voice"] = str(cfg.get("speech_voice", ""))[:200]
        cfg["vision_base_url"] = str(cfg.get("vision_base_url", "")).strip()
        cfg["vision_model"] = str(cfg.get("vision_model", "")).strip()
        cfg["player_name"] = str(cfg.get("player_name", "老哥")).strip()[:12] or "老哥"
        try:
            cfg["screen_interval"] = max(2, min(30, int(cfg.get("screen_interval", 5))))
        except (TypeError, ValueError):
            cfg["screen_interval"] = 5
        cfg["screen_speak"] = bool(cfg.get("screen_speak", True))
        cfg["screen_skip_static"] = bool(cfg.get("screen_skip_static", True))
        cfg["wake_words"] = normalize_wake_words(cfg.get("wake_words"))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(CONFIG_PATH)
        return cfg


def public_config(cfg: dict | None = None) -> dict:
    """返回给前端的配置：不下发明文 API Key。"""
    cfg = copy.deepcopy(cfg or load_config())
    key = str(cfg.pop("api_key", "") or "")
    vision_key = str(cfg.pop("vision_api_key", "") or "")
    cfg["has_api_key"] = bool(key)
    cfg["api_key_masked"] = (key[:4] + "****" + key[-4:]) if len(key) > 8 else ("****" if key else "")
    cfg["has_vision_key"] = bool(vision_key or key)
    cfg["vision_api_key_masked"] = (
        (vision_key[:4] + "****" + vision_key[-4:]) if len(vision_key) > 8 else ("****" if vision_key else "")
    )
    cfg["vision_ready"] = bool(cfg.get("vision_model")) and cfg["has_vision_key"]
    return cfg
