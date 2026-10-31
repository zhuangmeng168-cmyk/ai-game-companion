"""联网大模型接入：OpenAI 兼容协议（DeepSeek / OpenAI / 任意兼容网关）。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """模型调用失败，上层会把错误原样提示给玩家。"""


def chat(cfg: dict, messages: list[dict], timeout: float = 25.0) -> str:
    base = str(cfg.get("base_url") or "https://api.deepseek.com/v1").strip().rstrip("/")
    key = str(cfg.get("api_key") or "").strip()
    if not key:
        raise LLMError("还没填 API Key，去右上角 ⚙ 设置里填一个")
    if base.endswith("/chat/completions"):
        url = base
    else:
        url = base + "/chat/completions"
    payload = {
        "model": cfg.get("model") or "deepseek-chat",
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.9)),
        "max_tokens": 64,
        "stream": False,
    }
    data = post_json(url, payload, {"Authorization": f"Bearer {key}"}, timeout)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("模型返回格式不对，检查接口地址和模型名") from exc
    return str(content or "").strip()


def post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 25.0) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        if exc.code in (401, 403):
            raise LLMError("API Key 不对或没有权限，去设置里检查一下") from exc
        if exc.code == 429:
            raise LLMError("请求太频繁或额度用完了，等会儿再试") from exc
        raise LLMError(f"模型接口返回 {exc.code}：{detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"连不上模型接口，检查网络或接口地址（{exc}）") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型返回的不是 JSON：{raw[:120]}") from exc
    if not isinstance(data, dict):
        raise LLMError("模型返回格式不对")
    return data
