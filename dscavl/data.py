"""DSCA 格式数据：视频/问题索引、字幕加载、``QuestionFeatureDataset`` 与 collate 函数。"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class QuestionSample:
    """单条 MCQ/QA 样本及对应视频、字幕、特征路径元数据。"""
    video_id: str
    question_id: str
    question: str
    options: List[str]
    answer: str
    task_type: str
    duration: str
    domain: str
    sub_category: str
    video_path: str
    subtitle_path: Optional[str]
    feature_path: Optional[str]
    subtitle_text: Optional[str] = None


@dataclass
class VideoSample:
    """单个视频及其下挂的所有 ``QuestionSample``。"""
    video_id: str
    video_path: str
    subtitle_path: Optional[str]
    feature_path: Optional[str]
    questions: List[QuestionSample]
    subtitle_text: Optional[str] = None


def _clean_subtitle_text(text: str) -> str:
    """去掉 SRT 标签、序号与时间轴行，合并为单行文本。"""
    text = _TAG_RE.sub("", text)
    lines = [line.strip() for line in text.splitlines()]
    kept = []
    for line in lines:
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        kept.append(line)
    return " ".join(kept)


def load_subtitle_text(subtitle_path: Optional[str]) -> Optional[str]:
    """读取字幕文件并清洗；不存在则返回 None。"""
    if subtitle_path is None:
        return None
    path = Path(subtitle_path)
    if not path.exists():
        return None
    return _clean_subtitle_text(path.read_text(encoding="utf-8", errors="ignore"))


def load_query_records(query_path: str) -> List[Dict[str, Any]]:
    """读取 ``query/{video_id}.json`` 中的问题列表。"""
    with open(query_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_paths(root: Path, video_id: str, feature_dir: Optional[Path]) -> Dict[str, Optional[Path]]:
    """解析视频、字幕、特征缓存的标准相对路径。"""
    video_path = root / "video" / f"{video_id}.mp4"
    subtitle_path = root / "subtitle" / f"{video_id}.srt"
    feature_path = feature_dir / f"{video_id}.pt" if feature_dir is not None else None
    return {
        "video_path": video_path,
        "subtitle_path": subtitle_path,
        "feature_path": feature_path,
    }


def _build_question_samples(
    video_id: str,
    questions_raw: List[Dict[str, Any]],
    paths: Dict[str, Optional[Path]],
    subtitle_text: Optional[str],
) -> List[QuestionSample]:
    """将 JSON 字段映射为强类型 ``QuestionSample`` 列表。"""
    questions: List[QuestionSample] = []
    subtitle_path = paths["subtitle_path"]
    feature_path = paths["feature_path"]
    video_path = paths["video_path"]

    for item in questions_raw:
        questions.append(
            QuestionSample(
                video_id=video_id,
                question_id=item["question_id"],
                question=item["question"],
                options=item.get("options", []),
                answer=item.get("answer", ""),
                task_type=item.get("task_type", ""),
                duration=item.get("duration", ""),
                domain=item.get("domain", ""),
                sub_category=item.get("sub_category", ""),
                video_path=str(video_path),
                subtitle_path=str(subtitle_path) if subtitle_path is not None and subtitle_path.exists() else None,
                feature_path=str(feature_path) if feature_path is not None else None,
                subtitle_text=subtitle_text,
            )
        )

    return questions


def build_video_index(data_root: str, feature_root: Optional[str] = None, include_subtitles: bool = False) -> List[VideoSample]:
    """扫描 ``data_root/query/*.json`` 构建全库 ``VideoSample`` 索引。"""
    root = Path(data_root)
    query_dir = root / "query"
    feature_dir = Path(feature_root) if feature_root is not None else None

    samples: List[VideoSample] = []

    for query_path in sorted(query_dir.glob("*.json")):
        video_id = query_path.stem
        paths = _resolve_paths(root, video_id, feature_dir)
        questions_raw = load_query_records(str(query_path))
        subtitle_path = paths["subtitle_path"]
        subtitle_text = None
        if include_subtitles and subtitle_path is not None and subtitle_path.exists():
            subtitle_text = load_subtitle_text(str(subtitle_path))

        questions = _build_question_samples(video_id, questions_raw, paths, subtitle_text)

        samples.append(
            VideoSample(
                video_id=video_id,
                video_path=str(paths["video_path"]),
                subtitle_path=str(subtitle_path) if subtitle_path is not None and subtitle_path.exists() else None,
                feature_path=str(paths["feature_path"]) if paths["feature_path"] is not None else None,
                questions=questions,
                subtitle_text=subtitle_text,
            )
        )

    return samples


def flatten_questions(video_samples: List[VideoSample]) -> List[QuestionSample]:
    """将按视频组织的数据摊平为按问题遍历（Dataset 常用）。"""
    questions: List[QuestionSample] = []
    for sample in video_samples:
        questions.extend(sample.questions)
    return questions


def single_sample_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``batch_size=1`` 时直接返回单条 dict，避免多余维。"""
    if len(batch) != 1:
        raise ValueError("single_sample_collate expects DataLoader batch_size=1")
    return batch[0]


def variable_feature_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """列表字段按 batch 收集；``input_ids``/``attention_mask`` 堆叠为张量。"""
    if not batch:
        raise ValueError("Received empty batch")

    out: Dict[str, Any] = {
        "video_id": [item["video_id"] for item in batch],
        "question_id": [item["question_id"] for item in batch],
        "question": [item["question"] for item in batch],
        "options": [item["options"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "task_type": [item["task_type"] for item in batch],
        "duration": [item.get("duration", "") for item in batch],
        "domain": [item.get("domain", "") for item in batch],
        "sub_category": [item.get("sub_category", "") for item in batch],
        "video_path": [item["video_path"] for item in batch],
        "subtitle_path": [item["subtitle_path"] for item in batch],
        "feature_path": [item["feature_path"] for item in batch],
        "subtitle_text": [item["subtitle_text"] for item in batch],
        "precomputed_features": [item.get("precomputed_features") for item in batch],
        "timestamps": [item.get("timestamps") for item in batch],
        "cache_version": [item.get("cache_version") for item in batch],
        "feature_layout": [item.get("feature_layout") for item in batch],
        "patch_tokens_per_frame": [item.get("patch_tokens_per_frame") for item in batch],
        "patch_grid_size": [item.get("patch_grid_size") for item in batch],
        "source_patch_grid_size": [item.get("source_patch_grid_size") for item in batch],
    }

    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([item["input_ids"] for item in batch], dim=0)
    if "attention_mask" in batch[0]:
        out["attention_mask"] = torch.stack([item["attention_mask"] for item in batch], dim=0)

    return out


class QuestionFeatureDataset(Dataset):
    """每条样本可选加载 ``.pt`` 预计算特征与 tokenizer 编码后的 query。"""

    def __init__(
        self,
        data_root: str,
        feature_root: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
        include_subtitles: bool = False,
    ):
        """``include_subtitles=True`` 时在索引阶段读入字幕文本供弱监督使用。"""
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.video_samples = build_video_index(
            data_root=data_root,
            feature_root=feature_root,
            include_subtitles=include_subtitles,
        )
        self.samples = flatten_questions(self.video_samples)

    def __len__(self) -> int:
        """数据集大小等于问题条数。"""
        return len(self.samples)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """将拼接了选项的题干编码为定长张量。"""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required when tokenized query tensors are requested")
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """返回元数据 + 可选 ``precomputed_features`` 与 ``input_ids`` 等。"""
        sample = self.samples[index]
        item: Dict[str, Any] = {
            "video_id": sample.video_id,
            "question_id": sample.question_id,
            "question": sample.question,
            "options": sample.options,
            "answer": sample.answer,
            "task_type": sample.task_type,
            "duration": sample.duration,
            "domain": sample.domain,
            "sub_category": sample.sub_category,
            "video_path": sample.video_path,
            "subtitle_path": sample.subtitle_path,
            "feature_path": sample.feature_path,
            "subtitle_text": sample.subtitle_text,
        }

        if sample.feature_path is not None and Path(sample.feature_path).exists():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*weights_only.*",
                    category=FutureWarning,
                )
                payload = torch.load(sample.feature_path, map_location="cpu")
            item["precomputed_features"] = payload["features"]
            item["timestamps"] = payload.get("timestamps")
            item["cache_version"] = payload.get("cache_version")
            item["feature_layout"] = payload.get("feature_layout")
            item["patch_tokens_per_frame"] = payload.get("patch_tokens_per_frame")
            item["patch_grid_size"] = payload.get("patch_grid_size")
            item["source_patch_grid_size"] = payload.get("source_patch_grid_size")

        if self.tokenizer is not None:
            prompt = sample.question
            if sample.options:
                prompt = prompt + " Options: " + " ".join(sample.options)
            item.update(self._tokenize(prompt))

        return item
