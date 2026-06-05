from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

from ..models.schemas import DocumentChunk


class SimpleBM25Index:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.doc_freq = Counter()
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.avgdl = 0.0
        self._build()

    def search(self, query: str, top_k: int) -> list[tuple[DocumentChunk, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_term_freq = Counter(query_tokens)
        scores: dict[int, float] = defaultdict(float)

        for token, query_tf in query_term_freq.items():
            postings = self.postings.get(token)
            if not postings:
                continue

            idf = math.log(1 + (len(self.chunks) - self.doc_freq[token] + 0.5) / (self.doc_freq[token] + 0.5))
            for doc_index, term_tf in postings:
                doc_len = self.doc_lengths[doc_index] or 1
                numerator = term_tf * 2.2
                denominator = term_tf + 1.2 * (1 - 0.75 + 0.75 * doc_len / (self.avgdl or 1.0))
                scores[doc_index] += idf * (numerator / denominator) * query_tf

        if not scores:
            return []

        top_hits = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [(self.chunks[doc_index], score) for doc_index, score in top_hits if score > 0]

    def _build(self) -> None:
        total_length = 0
        for doc_index, chunk in enumerate(self.chunks):
            tokens = self._tokenize(chunk.text)
            counts = Counter(tokens)
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            total_length += doc_length
            for term, term_tf in counts.items():
                self.doc_freq[term] += 1
                self.postings[term].append((doc_index, term_tf))
        self.avgdl = total_length / max(len(self.chunks), 1)

    def _tokenize(self, text: str) -> list[str]:
        normalized = text or ""
        tokens: list[str] = []

        for part in re.findall(r"[A-Za-z0-9]{1,20}", normalized):
            tokens.append(part)

        for part in re.findall(r"[\u4e00-\u9fa5]{2,}", normalized):
            tokens.append(part)
            for size in range(2, min(5, len(part) + 1)):
                for start in range(0, len(part) - size + 1):
                    tokens.append(part[start : start + size])

        return tokens
