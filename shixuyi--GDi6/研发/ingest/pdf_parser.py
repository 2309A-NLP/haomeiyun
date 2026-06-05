from __future__ import annotations

import re
from pathlib import Path

import fitz

from ..core.logging import logger
from ..models.schemas import DocumentChunk


CHAPTER_RE = re.compile(r"^第[0-9\u4e00-\u9fff]+节")
LIST_MARKER_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.、)]|[(（]?[0-9]+[)）]|[一二三四五六七八九十百]+[、.)）]|[-*•])"
)
SENTENCE_END_RE = re.compile(r"[。！？!?；;】\]）)]$")
STRONG_PARAGRAPH_STARTERS = (
    "公司",
    "发行人",
    "报告期",
    "另外",
    "此外",
    "同时",
    "但是",
    "因此",
    "截至",
    "根据",
    "未来",
    "目前",
)


class ProspectusPDFParser:
    def parse(self, pdf_path: Path, source_name: str) -> list[dict]:
        pdf_path = Path(pdf_path)
        if pdf_path.suffix.lower() == ".txt":
            paragraphs = self._parse_processed_text(pdf_path, source_name)
        else:
            paragraphs = self._parse_pdf(pdf_path, source_name)

        paragraphs = self._repair_paragraphs(paragraphs)
        logger.info("Parsed %s paragraph chunks from %s", len(paragraphs), pdf_path)
        return paragraphs

    def chunk_pages(self, pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
        del chunk_size
        del chunk_overlap

        chunks: list[DocumentChunk] = []
        for page_data in pages:
            paragraph_text = self._normalize_text(page_data["text"])
            if not paragraph_text:
                continue

            paragraph_index = page_data.get("paragraph_index", 0)
            chunks.append(
                DocumentChunk(
                    chunk_id=f'{page_data["page"]}-p{paragraph_index}',
                    source=page_data["source"],
                    page=page_data["page"],
                    text=paragraph_text,
                    title=page_data["title"],
                    keywords=self._extract_keywords(paragraph_text),
                    metadata={
                        "chunking_strategy": "paragraph",
                        "paragraph_index": paragraph_index,
                    },
                )
            )

        logger.info("Built %s paragraph-level chunks", len(chunks))
        return chunks

    def _parse_pdf(self, pdf_path: Path, source_name: str) -> list[dict]:
        document = fitz.open(pdf_path)
        paragraphs: list[dict] = []
        try:
            for page_index, page in enumerate(document, start=1):
                page_paragraphs = self._extract_page_paragraphs(page)
                for paragraph_index, paragraph_text in enumerate(page_paragraphs):
                    if (
                        paragraphs
                        and self._should_merge_across_pages(
                            previous=paragraphs[-1],
                            current_text=paragraph_text,
                            current_page=page_index,
                        )
                    ):
                        paragraphs[-1]["text"] = self._join_paragraph_text(paragraphs[-1]["text"], paragraph_text)
                        continue

                    paragraphs.append(
                        {
                            "page": page_index,
                            "source": source_name,
                            "title": source_name,
                            "paragraph_index": paragraph_index,
                            "text": paragraph_text,
                        }
                    )
        finally:
            document.close()
        return paragraphs

    def _parse_processed_text(self, text_path: Path, source_name: str) -> list[dict]:
        paragraphs: list[dict] = []
        current_page = 1
        page_paragraph_index = 0

        for raw_line in text_path.read_text(encoding="utf-8").splitlines():
            line = self._normalize_text(raw_line)
            if not line:
                continue

            page_match = re.match(r"^\[PAGE\s+(\d+)\]$", line)
            if page_match:
                current_page = int(page_match.group(1))
                page_paragraph_index = 0
                continue

            if (
                paragraphs
                and self._should_merge_across_pages(
                    previous=paragraphs[-1],
                    current_text=line,
                    current_page=current_page,
                )
            ):
                paragraphs[-1]["text"] = self._join_paragraph_text(paragraphs[-1]["text"], line)
                continue

            paragraphs.append(
                {
                    "page": current_page,
                    "source": source_name,
                    "title": source_name,
                    "paragraph_index": page_paragraph_index,
                    "text": line,
                }
            )
            page_paragraph_index += 1

        return paragraphs

    def _extract_page_paragraphs(self, page: fitz.Page) -> list[str]:
        blocks = page.get_text("blocks") or []
        candidates: list[tuple[float, float, float, float, str]] = []
        for block in blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            normalized = self._normalize_text(text)
            if not normalized or self._should_skip_block(normalized):
                continue
            candidates.append((round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2), normalized))

        candidates.sort(key=lambda item: (item[1], item[0]))
        return self._merge_blocks_to_paragraphs(candidates)

    def _merge_blocks_to_paragraphs(self, blocks: list[tuple[float, float, float, float, str]]) -> list[str]:
        paragraphs: list[str] = []
        current_lines: list[str] = []
        prev_x0: float | None = None
        prev_y1: float | None = None
        prev_text = ""

        for x0, y0, _x1, y1, text in blocks:
            raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not raw_lines:
                continue

            for line in raw_lines:
                starts_new = self._is_new_paragraph(
                    current_lines=current_lines,
                    prev_x0=prev_x0,
                    prev_y1=prev_y1,
                    prev_text=prev_text,
                    x0=x0,
                    y0=y0,
                    text=line,
                )
                if starts_new and current_lines:
                    merged = self._merge_wrapped_lines("\n".join(current_lines))
                    if merged:
                        paragraphs.append(merged)
                    current_lines = [line]
                else:
                    current_lines.append(line)

                prev_x0 = x0
                prev_y1 = y1
                prev_text = line

        if current_lines:
            merged = self._merge_wrapped_lines("\n".join(current_lines))
            if merged:
                paragraphs.append(merged)
        return paragraphs

    def _is_new_paragraph(
        self,
        current_lines: list[str],
        prev_x0: float | None,
        prev_y1: float | None,
        prev_text: str,
        x0: float,
        y0: float,
        text: str,
    ) -> bool:
        if not current_lines:
            return True

        if self._should_start_new_paragraph(text):
            return True

        if prev_y1 is not None:
            vertical_gap = y0 - prev_y1
            if vertical_gap > 18:
                return True

        if prev_x0 is not None and abs(x0 - prev_x0) > 18 and self._looks_like_indented_start(text):
            return True

        if self._ends_with_open_clause(prev_text) and self._looks_like_indented_start(text):
            return True

        return False

    def _merge_wrapped_lines(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        merged: list[str] = []
        for line in lines:
            if not merged:
                merged.append(line)
                continue

            if self._should_start_new_paragraph(line):
                merged.append("\n" + line)
            elif self._should_join_without_space(merged[-1], line):
                merged[-1] = f"{merged[-1]}{line}"
            else:
                merged[-1] = f"{merged[-1]} {line}"

        compact = "\n".join(part.strip() for part in "".join(merged).split("\n") if part.strip())
        return compact.strip()

    def _repair_paragraphs(self, paragraphs: list[dict]) -> list[dict]:
        if not paragraphs:
            return []

        repaired: list[dict] = [paragraphs[0].copy()]
        for item in paragraphs[1:]:
            current = item.copy()
            if self._should_merge_adjacent_paragraphs(repaired[-1], current):
                repaired[-1]["text"] = self._join_paragraph_text(repaired[-1]["text"], current["text"])
                continue
            repaired.append(current)

        return self._reindex_paragraphs(repaired)

    def _should_merge_adjacent_paragraphs(self, previous: dict, current: dict) -> bool:
        previous_page = int(previous.get("page") or 0)
        current_page = int(current.get("page") or 0)
        if current_page not in {previous_page, previous_page + 1}:
            return False

        previous_text = self._normalize_text(previous.get("text", ""))
        current_text = self._normalize_text(current.get("text", ""))
        if not previous_text or not current_text:
            return False

        if self._looks_like_standalone_title(current_text):
            return False

        if self._should_merge_across_pages(previous, current_text, current_page):
            return True

        if previous_page != current_page:
            return False

        if self._is_table_label(previous_text) or self._is_table_label(current_text):
            return False

        if len(current_text) <= 12:
            return True

        if len(previous_text) <= 8 and not self._looks_like_standalone_title(previous_text):
            return True

        if self._ends_with_open_clause(previous_text):
            return True

        if self._starts_with_continuation(current_text):
            return True

        if not self._ends_sentence(previous_text):
            if len(previous_text) <= 24 or len(current_text) <= 24:
                return True
            if self._looks_like_mid_sentence(previous_text, current_text):
                return True
            if self._looks_like_wrapped_continuation(previous_text, current_text):
                return True

        return False

    def _looks_like_wrapped_continuation(self, previous_text: str, current_text: str) -> bool:
        if self._looks_like_new_paragraph_start(current_text):
            return False
        if self._ends_with_open_clause(previous_text):
            return True
        if re.match(r"^[\u4e00-\u9fa5]", current_text):
            return True
        return False

    def _looks_like_new_paragraph_start(self, text: str) -> bool:
        compact = text.strip()
        if not compact:
            return False
        if self._looks_like_standalone_title(compact) or self._is_table_label(compact):
            return True
        if self._should_start_new_paragraph(compact):
            return True
        return compact.startswith(STRONG_PARAGRAPH_STARTERS)

    def _looks_like_unfinished_tail(self, text: str) -> bool:
        compact = text.replace(" ", "")
        if not compact or self._ends_sentence(compact):
            return False
        if self._ends_with_open_clause(compact):
            return True
        if len(compact) <= 40:
            return True
        return bool(
            re.search(
                r"(?:第[一二三四五六七八九十百千万0-9]*个?|前[一二三四五六七八九十百千万0-9]*|"
                r"用户|系统|平台|设备|产品|技术|标准|项目|方案|模块|领域|能力|需求|"
                r"金额|比例|收入|利润|市场|客户|用途|情形|如下|包括|采用|用于|形成|满足|实现|"
                r"某|该|本|其)$",
                compact,
            )
        )

    def _join_paragraph_text(self, previous_text: str, current_text: str) -> str:
        previous_text = self._normalize_text(previous_text)
        current_text = self._normalize_text(current_text)
        if not previous_text:
            return current_text
        if not current_text:
            return previous_text
        if self._should_join_without_space(previous_text, current_text):
            return f"{previous_text}{current_text}"
        return f"{previous_text} {current_text}"

    def _reindex_paragraphs(self, paragraphs: list[dict]) -> list[dict]:
        counters: dict[int, int] = {}
        for item in paragraphs:
            page = int(item.get("page") or 0)
            counters.setdefault(page, 0)
            item["paragraph_index"] = counters[page]
            counters[page] += 1
        return paragraphs

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"\r\n?", "\n", text or "")
        normalized = re.sub(r"[ \t\u3000]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _should_skip_block(self, text: str) -> bool:
        compact = text.replace(" ", "")
        if not compact:
            return True
        if re.fullmatch(r"\d+-\d+-\d+", compact):
            return True
        if re.fullmatch(r"\d{1,4}", compact):
            return True
        if len(compact) <= 2:
            return True
        return False

    def _looks_like_standalone_title(self, text: str) -> bool:
        compact = text.replace(" ", "")
        if CHAPTER_RE.match(compact):
            return True
        if re.match(r"^[一二三四五六七八九十]+、", compact):
            return True
        if re.match(r"^[(（][一二三四五六七八九十0-9]+[)）]", compact):
            return True
        if re.match(r"^\d+[、.)）]", compact):
            return True
        return len(compact) <= 20 and compact.endswith(
            ("风险", "概况", "释义", "情况", "用途", "概算", "模式", "规划", "前景", "格局")
        )

    def _is_table_label(self, text: str) -> bool:
        compact = text.replace(" ", "")
        if compact.startswith(("单位：", "单位:")):
            return True
        return compact in {"项目", "类别", "图例", "简介", "营业利润", "变动率", "发行费用概算"}

    def _ends_with_open_clause(self, text: str) -> bool:
        return text.endswith(("、", "，", ",", "：", ":", "（", "(", "【", "[", "%", "‰", "的", "与", "及"))

    def _starts_with_continuation(self, text: str) -> bool:
        if re.match(r"^[0-9A-Za-z%％\-\+\./,，。；;:：)）\]】]", text):
            return True
        return text.startswith(("及", "和", "与", "其中", "以及", "分别", "占", "为", "的", "并", "且"))

    def _ends_sentence(self, text: str) -> bool:
        return bool(SENTENCE_END_RE.search(text))

    def _looks_like_mid_sentence(self, previous_text: str, current_text: str) -> bool:
        if current_text.startswith(("公司", "发行人", "报告期内", "其中", "此外", "并且")):
            return True
        return bool(
            re.search(r"[A-Za-z0-9%]$", previous_text)
            and re.match(r"^[A-Za-z0-9%]", current_text)
        )

    def _should_start_new_paragraph(self, line: str) -> bool:
        compact = line.replace(" ", "")
        return bool(CHAPTER_RE.match(compact) or LIST_MARKER_RE.match(compact))

    def _looks_like_indented_start(self, text: str) -> bool:
        compact = text.replace(" ", "")
        return bool(LIST_MARKER_RE.match(compact))

    def _should_merge_across_pages(self, previous: dict, current_text: str, current_page: int) -> bool:
        previous_text = self._normalize_text(previous.get("text", ""))
        if previous.get("page") != current_page - 1:
            return False
        if not previous_text or not current_text:
            return False
        if self._looks_like_standalone_title(current_text):
            return False
        if self._ends_sentence(previous_text):
            return False
        return True

    def _should_join_without_space(self, previous: str, current: str) -> bool:
        if previous.endswith(("(", "[", "{", '"', "“", ":", "：", "（")):
            return True
        if current.startswith((")", "]", "}", '"', "”", ",", ".", ";", ":", "，", "。", "；", "：", "）")):
            return True
        if self._looks_like_standalone_title(previous.strip()) or self._should_start_new_paragraph(previous.strip()):
            return False
        prev_char = previous.rstrip()[-1] if previous.rstrip() else ""
        curr_char = current.lstrip()[0] if current.lstrip() else ""
        if self._is_cjk_char(prev_char) and self._is_cjk_char(curr_char):
            return True
        return bool(re.match(r"^[A-Za-z0-9%]", current) and re.search(r"[A-Za-z0-9]$", previous))

    def _is_cjk_char(self, char: str) -> bool:
        return bool(char and "\u4e00" <= char <= "\u9fff")

    def _extract_keywords(self, text: str) -> list[str]:
        candidates = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}", text)
        seen: set[str] = set()
        keywords: list[str] = []
        for token in candidates:
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= 18:
                break
        return keywords
