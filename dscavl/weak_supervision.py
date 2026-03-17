from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


@dataclass
class SubtitleSegment:
    start_sec: float
    end_sec: float
    text: str


_WORD_RE = re.compile(r"[a-z0-9']+")
_TIMING_RE = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s+-->\s+(\d\d):(\d\d):(\d\d),(\d\d\d)")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "in", "is", "it", "its", "of", "on", "or", "the", "this", "that", "to", "was", "what", "which",
    "who", "why", "with", "video", "about", "into", "today", "end",
}


def _to_seconds(hh: str, mm: str, ss: str, ms: str) -> float:
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt_segments(subtitle_path: str | Path) -> list[SubtitleSegment]:
    path = Path(subtitle_path)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    segments: list[SubtitleSegment] = []
    idx = 0
    while idx < len(lines):
        timing = _TIMING_RE.search(lines[idx])
        if timing is None:
            idx += 1
            continue

        start_sec = _to_seconds(*timing.groups()[0:4])
        end_sec = _to_seconds(*timing.groups()[4:8])
        idx += 1
        text_lines = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(re.sub(r"<[^>]+>", "", lines[idx]).strip())
            idx += 1
        text = " ".join([line for line in text_lines if line])
        if text:
            segments.append(SubtitleSegment(start_sec=start_sec, end_sec=end_sec, text=text))
        idx += 1
    return segments


def extract_keywords(question: str, options: Sequence[str]) -> set[str]:
    text = " ".join([question, *options]).lower()
    words = _WORD_RE.findall(text)
    return {word for word in words if len(word) >= 3 and word not in _STOPWORDS and not word.isdigit()}


def match_subtitle_segments(segments: Iterable[SubtitleSegment], keywords: set[str], min_hits: int = 1) -> list[SubtitleSegment]:
    matched: list[SubtitleSegment] = []
    for segment in segments:
        segment_words = set(_WORD_RE.findall(segment.text.lower()))
        hits = len(segment_words & keywords)
        if hits >= min_hits:
            matched.append(segment)
    return matched


def build_gt_mask_from_subtitles(
    subtitle_path: str | None,
    timestamps: torch.Tensor | None,
    question: str,
    options: Sequence[str],
    min_hits: int = 1,
    expand_sec: float = 1.5,
) -> torch.Tensor | None:
    if subtitle_path is None or timestamps is None:
        return None

    segments = parse_srt_segments(subtitle_path)
    if not segments:
        return None

    keywords = extract_keywords(question, options)
    if not keywords:
        return None

    matched = match_subtitle_segments(segments, keywords, min_hits=min_hits)
    if not matched:
        return None

    times = timestamps.float()
    gt_mask = torch.zeros_like(times, dtype=torch.bool)
    for segment in matched:
        start = segment.start_sec - expand_sec
        end = segment.end_sec + expand_sec
        gt_mask |= (times >= start) & (times <= end)

    return gt_mask if gt_mask.any() else None