from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


Span = Tuple[str, int, int]


def normalize_span(file_name: str, start: int, end: int) -> Span:
    if start > end:
        start, end = end, start
    return file_name, int(start), int(end)


def merge_spans(spans: Iterable[Span]) -> List[Span]:
    grouped: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for file_name, start, end in spans:
        if start <= 0 or end <= 0:
            continue
        grouped[file_name].append((start, end))

    merged: List[Span] = []
    for file_name, intervals in grouped.items():
        intervals.sort()
        current_start = None
        current_end = None
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
                continue
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((file_name, current_start, current_end))
                current_start, current_end = start, end
        if current_start is not None:
            merged.append((file_name, current_start, current_end))
    return merged


def span_length(spans: Iterable[Span]) -> int:
    return sum(max(0, end - start + 1) for _, start, end in merge_spans(spans))


def overlap_length(left_spans: Iterable[Span], right_spans: Iterable[Span]) -> int:
    left_by_file: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    right_by_file: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for file_name, start, end in merge_spans(left_spans):
        left_by_file[file_name].append((start, end))
    for file_name, start, end in merge_spans(right_spans):
        right_by_file[file_name].append((start, end))

    total = 0
    for file_name, left_items in left_by_file.items():
        right_items = right_by_file.get(file_name, [])
        i = 0
        j = 0
        while i < len(left_items) and j < len(right_items):
            left_start, left_end = left_items[i]
            right_start, right_end = right_items[j]
            total += max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
            if left_end < right_end:
                i += 1
            else:
                j += 1
    return total


def docs_to_spans(docs: Iterable[Dict[str, Any]]) -> List[Span]:
    spans = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        file_name = metadata.get("file_name")
        start = metadata.get("global_start")
        end = metadata.get("global_end")
        if file_name and start is not None and end is not None:
            spans.append(normalize_span(str(file_name), int(start), int(end)))
    return spans


def evaluate_span_recall(
    gold_spans: Iterable[Span],
    retrieved_spans: Iterable[Span],
    hit_threshold: float = 0.5,
) -> Dict[str, float | bool]:
    gold = merge_spans(gold_spans)
    retrieved = merge_spans(retrieved_spans)

    gold_len = span_length(gold)
    retrieved_len = span_length(retrieved)
    overlap = overlap_length(gold, retrieved)
    union = gold_len + retrieved_len - overlap

    coverage = overlap / gold_len if gold_len else 0.0
    iou = overlap / union if union else 0.0

    per_gold_hits = []
    for gold_span in gold:
        gold_item_len = span_length([gold_span])
        item_overlap = overlap_length([gold_span], retrieved)
        per_gold_hits.append((item_overlap / gold_item_len) >= hit_threshold if gold_item_len else False)

    return {
        "gold_chars": gold_len,
        "retrieved_chars": retrieved_len,
        "overlap_chars": overlap,
        "coverage": coverage,
        "iou": iou,
        "hit": any(per_gold_hits),
        "span_recall": sum(1 for hit in per_gold_hits if hit) / len(per_gold_hits) if per_gold_hits else 0.0,
    }
