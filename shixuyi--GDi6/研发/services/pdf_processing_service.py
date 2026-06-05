from __future__ import annotations

import io
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from pdftext.extraction import dictionary_output
from rapidocr_onnxruntime import RapidOCR
from reportlab.lib.colors import black
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont, stringWidth
from reportlab.pdfgen import canvas

from ..core.config import settings
from ..core.logging import logger


PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s*\d+|\d+|[\-—_]*\s*\d+\s*[\-—_]*|\d+\s*/\s*\d+|\d+\-\d+\-\d+)$",
    re.IGNORECASE,
)
HEADING_RE = re.compile(
    r"^(第[0-9一二三四五六七八九十百千]+节|\d+(\.\d+)*[\.、)]|[(（]?(?:\d+|[一二三四五六七八九十百千]+)[)）]|[一二三四五六七八九十百千]+[、.)）])"
)
SENTENCE_END_RE = re.compile(r"[。！？；!?;:：]$")
TABLE_PREFIX_RE = re.compile(r"^(?:单位[:：]|项目(?:\s|$)|注[:：])")
DEFINITION_ROW_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z0-9《》（）()、，,\- ]{1,24}\s+指\s+.+$")
HEADING_BODY_SPLIT_RE = re.compile(
    r"^(?P<head>(?:第[0-9一二三四五六七八九十百千]+节|\d+(\.\d+)*[\.、)]|[(（]?\d+[)）]|[一二三四五六七八九十百千]+[、.)）])[^。！？；:：]{1,30})\s+(?P<body>.+)$"
)


@dataclass
class TextItem:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    source: str = "text"


@dataclass
class PageContent:
    page_number: int
    width: float
    height: float
    line_items: list[TextItem]
    paragraphs: list[str]


class PDFProcessingService:
    def __init__(self) -> None:
        self.ocr_engine: RapidOCR | None = None
        self._font_registered = False
        self._init_ocr_engine()

    def process(self, input_pdf: Path, output_pdf: Path, output_text: Path) -> dict:
        started = time.perf_counter()
        input_pdf = Path(input_pdf)
        output_pdf = Path(output_pdf)
        output_text = Path(output_text)

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_text.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Processing PDF with pdftext: input_pdf=%s output_pdf=%s output_text=%s",
            input_pdf,
            output_pdf,
            output_text,
        )

        extract_started = time.perf_counter()
        pdftext_pages = dictionary_output(str(input_pdf), sort=True, disable_links=True)
        extract_ms = int((time.perf_counter() - extract_started) * 1000)
        logger.info("PDF text extraction completed: input_pdf=%s pages=%s latency_ms=%s", input_pdf, len(pdftext_pages), extract_ms)

        build_started = time.perf_counter()
        with fitz.open(input_pdf) as fitz_pdf:
            repeated_rules = self._collect_repeated_line_rules(pdftext_pages)
            pages = self._build_page_contents(pdftext_pages, fitz_pdf, repeated_rules)
        build_ms = int((time.perf_counter() - build_started) * 1000)
        ocr_item_count = sum(1 for page in pages for item in page.line_items if item.source == "ocr")
        logger.info(
            "PDF page content built: input_pdf=%s page_count=%s repeated_rules=%s ocr_items=%s latency_ms=%s",
            input_pdf,
            len(pages),
            len(repeated_rules),
            ocr_item_count,
            build_ms,
        )

        write_text_started = time.perf_counter()
        self._write_clean_text(output_text, pages)
        write_text_ms = int((time.perf_counter() - write_text_started) * 1000)
        logger.info("Clean text written: output_text=%s latency_ms=%s", output_text, write_text_ms)

        write_pdf_started = time.perf_counter()
        self._write_clean_pdf(output_pdf, pages)
        write_pdf_ms = int((time.perf_counter() - write_pdf_started) * 1000)
        total_ms = int((time.perf_counter() - started) * 1000)

        stats = {
            "pages": len(pages),
            "paragraphs": sum(len(page.paragraphs) for page in pages),
            "lines": sum(len(page.line_items) for page in pages),
            "ocr_items": ocr_item_count,
            "repeated_rules": len(repeated_rules),
            "output_pdf": str(output_pdf),
            "output_text": str(output_text),
            "extract_ms": extract_ms,
            "build_ms": build_ms,
            "write_text_ms": write_text_ms,
            "write_pdf_ms": write_pdf_ms,
            "total_ms": total_ms,
        }
        logger.info("Finished PDF processing: %s", stats)
        return stats

    def render_pages_as_png_bytes(
        self,
        pdf_path: Path,
        page_numbers: list[int],
        scale: float | None = None,
    ) -> list[bytes]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return []

        scale = scale or settings.vlm_render_scale
        matrix = fitz.Matrix(scale, scale)
        rendered: list[bytes] = []
        logger.info(
            "Rendering PDF pages to PNG started: pdf_path=%s page_numbers=%s scale=%s",
            pdf_path,
            page_numbers,
            scale,
        )

        with fitz.open(pdf_path) as pdf:
            for page_number in page_numbers:
                page_index = page_number - 1
                if page_index < 0 or page_index >= len(pdf):
                    continue
                try:
                    pixmap = pdf[page_index].get_pixmap(matrix=matrix, alpha=False)
                    rendered.append(pixmap.tobytes("png"))
                except Exception as exc:  # pragma: no cover - runtime/pdf dependent
                    logger.warning("Failed to render page %s from %s: %s", page_number, pdf_path, exc)

        logger.info(
            "Rendering PDF pages to PNG completed: pdf_path=%s requested_pages=%s rendered_images=%s",
            pdf_path,
            len(page_numbers),
            len(rendered),
        )
        return rendered

    def ocr_pdf_region_text(
        self,
        pdf_path: Path,
        page_number: int,
        clip_ratios: tuple[float, float, float, float] | None = None,
        scale: float | None = None,
    ) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists() or self.ocr_engine is None:
            return ""

        started = time.perf_counter()
        scale = scale or settings.vlm_render_scale
        matrix = fitz.Matrix(scale, scale)
        logger.info(
            "OCR region extraction started: pdf_path=%s page_number=%s clip_ratios=%s scale=%s",
            pdf_path,
            page_number,
            clip_ratios,
            scale,
        )

        with fitz.open(pdf_path) as pdf:
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(pdf):
                return ""

            page = pdf[page_index]
            clip = None
            if clip_ratios is not None:
                left, top, right, bottom = clip_ratios
                rect = page.rect
                clip = fitz.Rect(
                    rect.x0 + rect.width * left,
                    rect.y0 + rect.height * top,
                    rect.x0 + rect.width * right,
                    rect.y0 + rect.height * bottom,
                )

            try:
                pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            except Exception as exc:  # pragma: no cover - runtime/pdf dependent
                logger.warning("Failed to render OCR region on page %s from %s: %s", page_number, pdf_path, exc)
                return ""

        text = self._ocr_image_bytes(pixmap.tobytes("png"))
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "OCR region extraction completed: pdf_path=%s page_number=%s text_length=%s latency_ms=%s",
            pdf_path,
            page_number,
            len(text),
            latency_ms,
        )
        return text

    def _init_ocr_engine(self) -> None:
        try:
            self.ocr_engine = RapidOCR()
        except Exception as exc:  # pragma: no cover - depends on local runtime
            self.ocr_engine = None
            logger.warning("RapidOCR init failed, OCR will be skipped: %s", exc)

    def _collect_repeated_line_rules(self, pdftext_pages: list[dict]) -> dict[tuple[str, str], int]:
        counts: Counter[tuple[str, str]] = Counter()
        total_pages = len(pdftext_pages)

        for page in pdftext_pages:
            height = float(page.get("height") or 0.0)
            seen: set[tuple[str, str]] = set()
            for line in self._extract_raw_lines(page):
                text = self._normalize_line(line.get("text", ""))
                if not text:
                    continue
                region = self._line_region(line, height)
                key = (text, region)
                if key in seen:
                    continue
                seen.add(key)
                counts[key] += 1

        threshold_top_bottom = max(3, math.ceil(total_pages * 0.18))
        threshold_center = max(4, math.ceil(total_pages * 0.30))

        rules: dict[tuple[str, str], int] = {}
        for key, count in counts.items():
            _, region = key
            threshold = threshold_center if region == "center" else threshold_top_bottom
            if count >= threshold:
                rules[key] = count

        logger.info("Detected %s repeated line rules", len(rules))
        return rules

    def _build_page_contents(
        self,
        pdftext_pages: list[dict],
        fitz_pdf: fitz.Document,
        repeated_rules: dict[tuple[str, str], int],
    ) -> list[PageContent]:
        pages: list[PageContent] = []

        for index, pdftext_page in enumerate(pdftext_pages, start=1):
            fitz_page = fitz_pdf[index - 1]
            width = float(fitz_page.rect.width)
            height = float(fitz_page.rect.height)

            line_items = self._extract_filtered_line_items(pdftext_page, repeated_rules)
            ocr_items = self._extract_image_ocr_items(fitz_page, line_items)
            all_items = sorted(line_items + ocr_items, key=lambda item: (round(item.top, 2), round(item.x0, 2)))
            if self._is_glossary_page(all_items):
                paragraphs = self._build_glossary_paragraphs(all_items)
            else:
                paragraphs = self._merge_items_to_paragraphs(all_items)

            pages.append(
                PageContent(
                    page_number=index,
                    width=width,
                    height=height,
                    line_items=all_items,
                    paragraphs=paragraphs,
                )
            )

        return pages

    def _is_glossary_page(self, items: list[TextItem]) -> bool:
        if not items:
            return False

        joined = "\n".join(item.text for item in items[:24])
        if not any(token in joined for token in ("基本用语", "专业术语", "具有如下含义")):
            return False

        definition_count = sum(1 for item in items if "指" in item.text)
        return definition_count >= 3

    def _build_glossary_paragraphs(self, items: list[TextItem]) -> list[str]:
        paragraphs: list[str] = []
        intro_markers = ("基本用语", "专业术语", "具有如下含义")
        glossary_started = not any(any(marker in item.text for marker in intro_markers) for item in items[:12]) and sum(
            1 for item in items[:12] if "指" in item.text
        ) >= 3
        split_x = self._glossary_split_x(items)
        term_parts: list[str] = []
        definition_parts: list[str] = []
        pending_definition_parts: list[str] = []

        def flush_entry() -> None:
            nonlocal term_parts, definition_parts
            if not term_parts and not definition_parts:
                return

            term_text = self._join_segments(term_parts)
            definition_text = self._join_segments(definition_parts)

            if term_text and definition_text:
                paragraphs.append(f"{term_text} 指 {definition_text}")
            elif term_text:
                paragraphs.append(term_text)
            elif definition_text:
                paragraphs.append(definition_text)

            term_parts = []
            definition_parts = []

        def start_entry(term_text: str, initial_definition: str = "") -> None:
            nonlocal term_parts, definition_parts, pending_definition_parts
            if term_parts and not definition_parts and not pending_definition_parts:
                term_parts.append(term_text)
            else:
                flush_entry()
                term_parts = [term_text] if term_text else []
            definition_parts = pending_definition_parts[:]
            pending_definition_parts = []
            if initial_definition:
                definition_parts.append(initial_definition)

        for index, item in enumerate(items):
            text = self._normalize_line(item.text)
            if not text:
                continue

            if any(token in text for token in intro_markers):
                flush_entry()
                if pending_definition_parts:
                    paragraphs.append(self._join_segments(pending_definition_parts))
                    pending_definition_parts = []
                paragraphs.append(text)
                glossary_started = True
                continue

            if not glossary_started:
                paragraphs.append(text)
                continue

            is_left = item.x0 < split_x
            normalized = re.sub(r"\s+", " ", text).strip()

            if is_left:
                if " 指 " in normalized:
                    left, right = normalized.split(" 指 ", 1)
                    start_entry(left.strip(), right.strip())
                elif normalized.endswith(" 指"):
                    start_entry(normalized[:-2].strip())
                else:
                    if term_parts and not definition_parts:
                        term_parts.append(normalized)
                    elif term_parts and definition_parts:
                        start_entry(normalized)
                    else:
                        term_parts = [normalized]
                continue

            if normalized == "指":
                continue
            if normalized.startswith("指 "):
                normalized = normalized[2:].strip()

            if term_parts:
                next_item = items[index + 1] if index + 1 < len(items) else None
                if (
                    next_item is not None
                    and next_item.x0 < split_x
                    and "指" in next_item.text
                    and 0 <= next_item.top - item.top <= 12
                ):
                    flush_entry()
                    pending_definition_parts.append(normalized)
                else:
                    definition_parts.append(normalized)
            else:
                pending_definition_parts.append(normalized)

        flush_entry()
        if pending_definition_parts:
            paragraphs.append(self._join_segments(pending_definition_parts))
        return [paragraph for paragraph in paragraphs if paragraph]

    def _glossary_split_x(self, items: list[TextItem]) -> float:
        left_positions = sorted(item.x0 for item in items if item.x0 <= 130)
        right_positions = sorted(item.x0 for item in items if item.x0 >= 160)
        if left_positions and right_positions:
            return (left_positions[0] + right_positions[0]) / 2.0
        return 150.0

    def _join_segments(self, segments: list[str]) -> str:
        merged = ""
        for segment in segments:
            part = self._normalize_line(segment)
            if not part:
                continue
            if not merged:
                merged = part
            elif self._join_without_space(merged, part):
                merged = f"{merged}{part}"
            else:
                merged = f"{merged} {part}"
        return merged.strip()

    def _extract_raw_lines(self, page: dict) -> list[dict]:
        blocks = page.get("blocks", []) if isinstance(page, dict) else []
        lines: list[dict] = []

        for block in blocks:
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                parts: list[str] = []
                x0_values: list[float] = []
                x1_values: list[float] = []
                top_values: list[float] = []
                bottom_values: list[float] = []

                for span in spans:
                    text = str(span.get("text", "")).strip()
                    bbox = span.get("bbox") or line.get("bbox") or block.get("bbox")
                    if not text or not bbox or len(bbox) < 4:
                        continue

                    parts.append(text)
                    x0_values.append(float(bbox[0]))
                    top_values.append(float(bbox[1]))
                    x1_values.append(float(bbox[2]))
                    bottom_values.append(float(bbox[3]))

                if not parts:
                    continue

                text = " ".join(parts)
                lines.append(
                    {
                        "text": text,
                        "x0": min(x0_values),
                        "x1": max(x1_values),
                        "top": min(top_values),
                        "bottom": max(bottom_values),
                    }
                )

        return sorted(lines, key=lambda item: (item["top"], item["x0"]))

    def _extract_filtered_line_items(
        self,
        page: dict,
        repeated_rules: dict[tuple[str, str], int],
    ) -> list[TextItem]:
        height = float(page.get("height") or 0.0)
        items: list[TextItem] = []

        for line in self._extract_raw_lines(page):
            text = self._normalize_line(line.get("text", ""))
            if not text:
                continue
            if self._should_skip_line(text, line, height, repeated_rules):
                continue

            items.append(
                TextItem(
                    text=text,
                    x0=float(line.get("x0", 0.0)),
                    top=float(line.get("top", 0.0)),
                    x1=float(line.get("x1", 0.0)),
                    bottom=float(line.get("bottom", 0.0)),
                    source="text",
                )
            )

        return items

    def _should_skip_line(
        self,
        text: str,
        line: dict,
        page_height: float,
        repeated_rules: dict[tuple[str, str], int],
    ) -> bool:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return True
        if PAGE_NUMBER_RE.fullmatch(compact):
            return True
        if len(compact) <= 1:
            return True

        region = self._line_region(line, page_height)
        if (text, region) in repeated_rules:
            return True

        if region == "center" and self._looks_like_watermark(text):
            return True
        return False

    def _line_region(self, line: dict, page_height: float) -> str:
        top = float(line.get("top", 0.0))
        bottom = float(line.get("bottom", top))
        middle = (top + bottom) / 2.0
        if top <= min(72.0, page_height * 0.12):
            return "top"
        if bottom >= max(page_height - 72.0, page_height * 0.88):
            return "bottom"
        if page_height and abs(middle - page_height / 2.0) <= page_height * 0.20:
            return "center"
        return "body"

    def _looks_like_watermark(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if len(compact) > 40:
            return False
        if any(token in compact.lower() for token in ("confidential", "draft", "sample")):
            return True
        if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9()（）\-_/]+", compact) and len(compact) >= 6:
            return True
        return False

    def _extract_image_ocr_items(self, page: fitz.Page, text_items: list[TextItem]) -> list[TextItem]:
        if self.ocr_engine is None:
            return []

        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
        items: list[TextItem] = []
        seen_texts: set[str] = set()
        max_images = 4

        for block in blocks:
            if block.get("type") != 1:
                continue
            bbox = block.get("bbox") or (0, 0, 0, 0)
            x0, y0, x1, y1 = [float(value) for value in bbox]
            width = x1 - x0
            height = y1 - y0
            area = width * height
            if width < 260 or height < 90 or area < 100000:
                continue
            if len(items) >= max_images:
                break

            image_bytes = block.get("image")
            if not image_bytes:
                continue

            text = self._ocr_image_bytes(image_bytes)
            text = self._normalize_line(text)
            compact = re.sub(r"\s+", "", text)
            if len(compact) < 6:
                continue
            if compact in seen_texts:
                continue
            if self._looks_like_duplicate_text(text, text_items):
                continue

            seen_texts.add(compact)
            items.append(TextItem(text=text, x0=x0, top=y0, x1=x1, bottom=y1, source="ocr"))

        return items

    def _ocr_image_bytes(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result, _ = self.ocr_engine(np.array(image))
        except Exception as exc:  # pragma: no cover - runtime/image dependent
            logger.warning("OCR failed on image block: %s", exc)
            return ""

        if not result:
            return ""

        parts: list[str] = []
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text = str(item[1]).strip()
            if text:
                parts.append(text)
        return " ".join(parts)

    def _looks_like_duplicate_text(self, ocr_text: str, text_items: list[TextItem]) -> bool:
        target = re.sub(r"\s+", "", ocr_text)
        for item in text_items:
            source = re.sub(r"\s+", "", item.text)
            if not source:
                continue
            if target in source or source in target:
                return True
        return False

    def _merge_items_to_paragraphs(self, items: list[TextItem]) -> list[str]:
        paragraphs: list[str] = []
        current: list[TextItem] = []

        for item in items:
            if not current:
                current = [item]
                continue

            if self._starts_new_paragraph(current[-1], item):
                paragraph = self._items_to_paragraph(current)
                if paragraph:
                    paragraphs.append(paragraph)
                current = [item]
            else:
                current.append(item)

        if current:
            paragraph = self._items_to_paragraph(current)
            if paragraph:
                paragraphs.append(paragraph)
        return self._refine_paragraphs(paragraphs)

    def _starts_new_paragraph(self, previous: TextItem, current: TextItem) -> bool:
        if previous.source != current.source:
            return True

        if self._looks_like_definition_item(previous.text) and self._looks_like_definition_term_start(current.text):
            return True

        if self._looks_like_definition_row(previous.text) or self._looks_like_definition_row(current.text):
            return True

        if self._looks_like_heading(previous.text) or self._looks_like_heading(current.text):
            return True

        if self._looks_like_table_row(previous.text) or self._looks_like_table_row(current.text):
            return True

        vertical_gap = current.top - previous.bottom
        prev_height = max(previous.bottom - previous.top, 1.0)
        if vertical_gap > max(10.0, prev_height * 1.6):
            return True

        if current.x0 - previous.x0 > 16:
            return True

        if HEADING_RE.match(current.text):
            return True

        if re.search(r"[。！？；!?;:]$", previous.text) and current.x0 <= previous.x0 + 6:
            return True

        return False

    def _items_to_paragraph(self, items: list[TextItem]) -> str:
        merged: list[str] = []
        for item in items:
            text = item.text.strip()
            if not text:
                continue
            if not merged:
                merged.append(text)
                continue

            table_context = self._looks_like_table_row(merged[-1]) or self._looks_like_table_row(text)
            if self._join_without_space(merged[-1], text, table_context=table_context):
                merged[-1] = f"{merged[-1]}{text}"
            else:
                merged[-1] = f"{merged[-1]} {text}"

        return " ".join(part.strip() for part in merged if part.strip()).strip()

    def _refine_paragraphs(self, paragraphs: list[str]) -> list[str]:
        refined: list[str] = []
        for paragraph in paragraphs:
            refined.extend(self._split_compound_paragraph(paragraph))
        refined = [item for item in refined if item]
        return self._merge_broken_paragraphs(refined)

    def _split_compound_paragraph(self, paragraph: str) -> list[str]:
        text = paragraph.strip()
        if not text:
            return []

        if self._looks_like_definition_row(text):
            return [text]

        unit_match = re.match(r"^(单位[:：][^ ]{0,12})\s+(.+)$", text)
        if unit_match:
            return [unit_match.group(1).strip(), unit_match.group(2).strip()]

        heading_match = HEADING_BODY_SPLIT_RE.match(text)
        if heading_match:
            head = heading_match.group("head").strip()
            body = heading_match.group("body").strip()
            if len(head.replace(" ", "")) <= 30 and len(body) >= 10:
                return [head, body]

        return [text]

    def _join_without_space(self, previous: str, current: str, table_context: bool = False) -> bool:
        if previous.endswith(("(", "[", "{", "“", '"', ":", "：")):
            return True
        if current.startswith((")", "]", "}", "”", '"', ",", ".", ";", ":", "，", "。", "；", "：")):
            return True
        if table_context:
            return False

        prev_char = previous.rstrip()[-1] if previous.rstrip() else ""
        curr_char = current.lstrip()[0] if current.lstrip() else ""
        if self._is_cjk_char(prev_char) and self._is_cjk_char(curr_char):
            return True
        if self._is_cjk_char(prev_char) and curr_char.isdigit():
            return True
        if prev_char.isdigit() and (self._is_cjk_char(curr_char) or curr_char in "%％"):
            return True
        if prev_char in "）)]】》”" and self._is_cjk_char(curr_char):
            return True

        if re.search(r"[A-Za-z0-9%]$", previous) and re.match(r"^[A-Za-z0-9%]", current):
            return True
        return False

    def _looks_like_heading(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if re.match(r"^\d+\.\d+[%％]", compact):
            return False
        return bool(HEADING_RE.match(compact) and len(compact) <= 40)

    def _looks_like_definition_row(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        return bool(DEFINITION_ROW_RE.match(normalized))

    def _looks_like_definition_item(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        return " 指 " in normalized and len(normalized) <= 80

    def _looks_like_definition_term_start(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized or " 指 " in normalized:
            return False
        if len(normalized) > 18:
            return False
        if self._ends_sentence(normalized):
            return False
        return bool(re.match(r"^[\u4e00-\u9fa5A-Za-z0-9《》（）()、,\- ]+$", normalized))

    def _looks_like_table_row(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        if TABLE_PREFIX_RE.match(text):
            return True
        parts = [part for part in text.split(" ") if part]
        numeric_groups = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", text)
        digit_parts = sum(any(char.isdigit() for char in part) for part in parts)
        if len(numeric_groups) >= 3 and len(parts) >= 4 and not self._ends_sentence(text):
            return True
        if len(parts) >= 6 and digit_parts >= 2 and not self._ends_sentence(text):
            return True
        return False

    def _ends_sentence(self, text: str) -> bool:
        return bool(SENTENCE_END_RE.search(text))

    def _is_cjk_char(self, char: str) -> bool:
        return bool(char and "\u4e00" <= char <= "\u9fff")

    def _normalize_line(self, text: str) -> str:
        normalized = re.sub(r"\r\n?", "\n", text or "")
        normalized = normalized.replace("\u00a0", " ")
        normalized = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", normalized)
        normalized = re.sub(r"[ \t\u3000]+", " ", normalized)
        normalized = re.sub(r"\n{2,}", " ", normalized)
        return normalized.strip()

    def _write_clean_text(self, output_path: Path, pages: list[PageContent]) -> None:
        merged_pages = self._prepare_text_pages(pages)
        parts: list[str] = []
        for page_number, page_paragraphs in merged_pages:
            parts.append(f"[PAGE {page_number}]")
            parts.extend(page_paragraphs)
            parts.append("")
        output_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")

    def _prepare_text_pages(self, pages: list[PageContent]) -> list[tuple[int, list[str]]]:
        prepared: list[tuple[int, list[str]]] = []

        for page in pages:
            paragraphs = list(page.paragraphs or [item.text for item in page.line_items])
            paragraphs = self._merge_broken_paragraphs(paragraphs)

            if prepared and paragraphs:
                previous_page, previous_paragraphs = prepared[-1]
                if previous_paragraphs and self._should_merge_broken_paragraph(previous_paragraphs[-1], paragraphs[0]):
                    previous_paragraphs[-1] = self._join_paragraph_segments(previous_paragraphs[-1], paragraphs[0])
                    paragraphs = paragraphs[1:]
                    prepared[-1] = (previous_page, previous_paragraphs)

            prepared.append((page.page_number, paragraphs))

        return prepared

    def _merge_broken_paragraphs(self, paragraphs: list[str]) -> list[str]:
        merged: list[str] = []
        for paragraph in paragraphs:
            text = paragraph.strip()
            if not text:
                continue

            if merged and self._should_merge_broken_paragraph(merged[-1], text):
                merged[-1] = self._join_paragraph_segments(merged[-1], text)
            else:
                merged.append(text)

        return merged

    def _should_merge_broken_paragraph(self, previous: str, current: str) -> bool:
        previous_text = previous.strip()
        current_text = current.strip()
        if not previous_text or not current_text:
            return False

        if any(self._looks_like_heading(text) for text in (previous_text, current_text)):
            return False
        if any(self._looks_like_definition_row(text) for text in (previous_text, current_text)):
            return False
        if self._is_strong_paragraph_continuation(previous_text, current_text):
            return True
        if any(self._looks_like_table_row(text) for text in (previous_text, current_text)):
            return False

        if current_text.startswith(("。", "，", "、", "；", "：", "）", "】", "%", "％")):
            return True
        if re.match(r"^(?:万元|元|年|月|日|次|项|股|天|%|％|[0-9])", current_text):
            return not self._ends_sentence(previous_text)
        if previous_text.endswith(("，", "、", "：", "（", "(", "与", "及", "和", "或", "的", "内")):
            return True
        if not self._ends_sentence(previous_text) and len(current_text) <= 20:
            return True
        return False

    def _is_strong_paragraph_continuation(self, previous: str, current: str) -> bool:
        if self._ends_sentence(previous):
            return False

        if current.startswith(("。", "，", "、", "；", "：", "）", "】", "%", "％")):
            return True
        if re.match(r"^(?:万元|元|年|月|日|次|项|股|天|名|个|%|％)", current):
            return True
        if previous.endswith(("，", "、", "：", "（", "(", "与", "及", "和", "或", "的", "内")):
            return True
        if re.search(r"(?:\d|[A-Za-z])$", previous) and re.match(r"^(?:\d|[A-Za-z]|万元|元|%|％)", current):
            return True
        return False

    def _join_paragraph_segments(self, previous: str, current: str) -> str:
        if self._join_without_space(previous, current):
            return f"{previous}{current}"
        return f"{previous} {current}"

    def _write_clean_pdf(self, output_path: Path, pages: list[PageContent]) -> None:
        self._register_font()
        pdf_canvas = canvas.Canvas(str(output_path))
        pdf_canvas.setTitle("Cleaned PDF")

        for page in pages:
            pdf_canvas.setPageSize((page.width, page.height))
            pdf_canvas.setFillColor(black)
            self._draw_page_paragraphs(pdf_canvas, page)

            pdf_canvas.showPage()

        pdf_canvas.save()

    def _register_font(self) -> None:
        if self._font_registered:
            return
        registerFont(UnicodeCIDFont("STSong-Light"))
        self._font_registered = True

    def _draw_text_item(self, pdf_canvas: canvas.Canvas, page: PageContent, item: TextItem) -> None:
        font_name = "STSong-Light"
        font_size = self._font_size_for_item(item)
        line_height = max(font_size * 1.25, 10)
        x = max(24.0, item.x0)
        top = max(18.0, item.top)
        width = max(80.0, min(item.x1 - item.x0, page.width - x - 24.0))
        y_top = page.height - top

        lines = self._wrap_text(item.text, width, font_name, font_size)
        for offset, line in enumerate(lines):
            y = y_top - offset * line_height
            if y <= 18:
                break
            pdf_canvas.setFont(font_name, font_size)
            pdf_canvas.drawString(x, y, line)

    def _draw_page_paragraphs(self, pdf_canvas: canvas.Canvas, page: PageContent) -> None:
        font_name = "STSong-Light"
        paragraphs = page.paragraphs or [item.text for item in page.line_items] or ["[Empty Page]"]
        left_margin = 32.0
        right_margin = 32.0
        top_margin = 28.0
        bottom_margin = 24.0
        available_width = max(120.0, page.width - left_margin - right_margin)
        available_height = max(120.0, page.height - top_margin - bottom_margin)

        font_size = self._fit_page_font_size(paragraphs, available_width, available_height, font_name)
        line_height = max(font_size * 1.45, 12.0)
        paragraph_gap = max(font_size * 0.55, 5.0)
        y = page.height - top_margin

        for paragraph in paragraphs:
            text = paragraph.strip()
            if not text:
                continue

            lines = self._wrap_text(text, available_width, font_name, font_size)
            for line in lines:
                if y <= bottom_margin:
                    return
                pdf_canvas.setFont(font_name, font_size)
                pdf_canvas.drawString(left_margin, y, line)
                y -= line_height
            y -= paragraph_gap

    def _fit_page_font_size(
        self,
        paragraphs: list[str],
        available_width: float,
        available_height: float,
        font_name: str,
    ) -> float:
        for font_size in (11.0, 10.5, 10.0, 9.5, 9.0, 8.5):
            line_height = max(font_size * 1.45, 12.0)
            paragraph_gap = max(font_size * 0.55, 5.0)
            used_height = 0.0
            for paragraph in paragraphs:
                lines = self._wrap_text(paragraph.strip(), available_width, font_name, font_size)
                used_height += len(lines) * line_height + paragraph_gap
            if used_height <= available_height:
                return font_size
        return 8.5

    def _font_size_for_item(self, item: TextItem) -> float:
        inferred = max(8.5, min(item.bottom - item.top + 1.5, 14.0))
        if item.source == "ocr":
            inferred = min(inferred, 10.5)
        return inferred

    def _wrap_text(self, text: str, max_width: float, font_name: str, font_size: float) -> list[str]:
        if not text:
            return [""]

        lines: list[str] = []
        current = ""
        for char in text:
            if char == "\n":
                if current:
                    lines.append(current)
                    current = ""
                continue
            trial = f"{current}{char}"
            if current and stringWidth(trial, font_name, font_size) > max_width:
                lines.append(current)
                current = char
            else:
                current = trial

        if current:
            lines.append(current)
        return lines or [text]
