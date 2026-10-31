"""Word（.docx）/ 纯文本攻略解析：只依赖 Python 标准库。

Word 的 .docx 本质是 zip，正文在 word/document.xml 里，这里提取：
- 段落文字（保留标题样式 → ## 章节）
- 项目符号（→ - 短句）
- 表格（按行转成「单元格1，单元格2」）
- 长段落里 3–10 字的短句自动变成「- 短句」，方便直接播报
"""

from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_STYLE = re.compile(r"(?:(?:heading|标题|h)\s*)?([1-6])", re.I)
_SENTENCE_SPLIT = re.compile(r"[。！？!?；;]+")


class DocumentError(ValueError):
    pass


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_document(filename: str, data: bytes) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".docx"):
        return extract_docx(data)
    if lower.endswith((".txt", ".md")):
        return decode_text(data)
    if lower.endswith(".doc"):
        raise DocumentError("旧版 .doc 格式不支持，请用 Word 另存为 .docx 再导入")
    raise DocumentError("只支持 .docx / .txt / .md 文档")


def extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            target = None
            if "word/document.xml" in archive.namelist():
                target = "word/document.xml"
            else:
                for name in archive.namelist():
                    if name.endswith("document.xml"):
                        target = name
                        break
            if target is None:
                raise DocumentError("这个 .docx 里没有找到正文")
            xml_bytes = archive.read(target)
    except zipfile.BadZipFile as exc:
        raise DocumentError("文件不是有效的 .docx（请确认没有改过扩展名）") from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DocumentError("Word 正文解析失败，文件可能损坏") from exc

    body = root.find(W + "body")
    if body is None:
        raise DocumentError("这个 .docx 里没有正文")
    lines: list[str] = []
    has_heading = False
    for element in body:
        if element.tag == W + "p":
            text = _paragraph_text(element)
            if not text:
                continue
            if _heading_level(element) is not None:
                lines.append("## " + text)
                has_heading = True
            elif _is_list(element):
                lines.append("- " + text)
            else:
                lines.extend(_convert_paragraph(text))
        elif element.tag == W + "tbl":
            lines.extend(_table_rows(element))
    if not lines:
        raise DocumentError("文档里没有可读取的文字（可能全是图片）")
    if not has_heading:
        lines = _auto_sections(lines)
    return "\n".join(lines)


def _paragraph_text(paragraph) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t":
            parts.append(node.text or "")
        elif tag == "tab":
            parts.append(" ")
        elif tag == "br":
            parts.append("\n")
    return "".join(parts).strip()


def _paragraph_style(paragraph) -> str:
    style = paragraph.find(f"{W}pPr/{W}pStyle")
    return (style.get(W + "val") if style is not None else "") or ""


def _heading_level(paragraph) -> int | None:
    style = _paragraph_style(paragraph).strip()
    match = _HEADING_STYLE.fullmatch(style)
    if match:
        return int(match.group(1))
    if style.lower() in ("title", "标题"):
        return 1
    outline = paragraph.find(f"{W}pPr/{W}outlineLvl")
    if outline is not None:
        try:
            return int(outline.get(W + "val")) + 1
        except (TypeError, ValueError):
            return None
    return None


def _is_list(paragraph) -> bool:
    return paragraph.find(f"{W}pPr/{W}numPr") is not None


def _convert_paragraph(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= 10:
        return ["- " + text]
    tips: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if 3 <= len(sentence) <= 10 and not sentence.startswith(("-", "•")):
            tips.append("- " + sentence)
    tips = list(dict.fromkeys(tips))[:3]
    return tips + [text] if tips else [text]


def _table_rows(table) -> list[str]:
    rows: list[str] = []
    for row in table.findall(W + "tr"):
        cells = [_paragraph_text(cell) for cell in row.findall(W + "tc")]
        cells = [cell for cell in cells if cell]
        if not cells:
            continue
        line = "，".join(cells)
        rows.append("- " + line if len(line) <= 10 else line)
    return rows


def _auto_sections(lines: list[str], max_chars: int = 800, max_lines: int = 10) -> list[str]:
    """没有标题的纯文本，按段落自动分节，避免整个文档变成一个大章节。"""
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        current.append(line)
        size += len(line)
        if size >= max_chars or len(current) >= max_lines:
            groups.append(current)
            current = []
            size = 0
    if current:
        groups.append(current)
    output: list[str] = []
    for index, group in enumerate(groups, 1):
        output.append(f"## 第{index}节")
        output.extend(group)
    return output
