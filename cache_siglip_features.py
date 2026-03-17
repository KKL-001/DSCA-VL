from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from dscavl.data import build_video_index


def _pick_first_tensor(source: Any, keys: Sequence[str]) -> Tuple[torch.Tensor | None, str | None]:
    for key in keys:
        value = None
        if isinstance(source, dict):
            value = source.get(key)
        elif hasattr(source, key):
            value = getattr(source, key)

        if value is not None:
            return value, key
    return None, None


def _extract_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    tensor, key = _pick_first_tensor(output, ("image_embeds", "pooler_output", "last_hidden_state"))
    if tensor is not None:
        return tensor.mean(dim=1) if key == "last_hidden_state" else tensor

    raise TypeError(f"Unsupported model output type for feature extraction: {type(output)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache SigLIP frame features for DSCA-VL videos.")
    parser.add_argument("--data-root", type=str, default="data", help="Directory containing query/, video/, subtitle/.")
    parser.add_argument("--feature-root", type=str, default="features", help="Directory to save cached frame features.")
    parser.add_argument("--model-name", type=str, default="google/siglip-base-patch16-224", help="Hugging Face SigLIP checkpoint.")
    parser.add_argument("--fps", type=float, default=1.0, help="Uniform video sampling rate in frames per second.")
    parser.add_argument("--batch-size", type=int, default=16, help="Frame batch size for encoder forward.")
    parser.add_argument("--max-frames", type=int, default=512, help="Max sampled frames per video; use <=0 to disable cap.")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto.")
    parser.add_argument("--use-slow-processor", action="store_true", help="Use slow image processor (disables fast processor warning).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute features even if cache file exists.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos for smoke testing.")
    parser.add_argument("--verbose", action="store_true", help="Print per-video progress details.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _downsample_timestamps(timestamps: List[float], max_frames: int) -> List[float]:
    if max_frames <= 0 or len(timestamps) <= max_frames:
        return timestamps
    if max_frames == 1:
        return [timestamps[0]]

    step = (len(timestamps) - 1) / (max_frames - 1)
    keep_idx = [int(round(i * step)) for i in range(max_frames)]
    return [timestamps[i] for i in keep_idx]


def sample_frames_uniform(video_path: str, fps: float, max_frames: int = 0) -> Tuple[List[Image.Image], List[float], float, int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0.0:
        native_fps = 30.0

    duration = frame_count / native_fps if frame_count > 0 else 0.0
    if duration <= 0.0:
        capture.release()
        raise RuntimeError(f"Invalid video duration: {video_path}")

    step = 1.0 / fps
    timestamps = []
    current = 0.0
    while current < duration:
        timestamps.append(current)
        current += step
    if not timestamps:
        timestamps = [0.0]
    timestamps = _downsample_timestamps(timestamps, max_frames)

    frames: List[Image.Image] = []
    actual_times: List[float] = []

    for ts in timestamps:
        frame_index = min(int(round(ts * native_fps)), max(frame_count - 1, 0))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        actual_times.append(frame_index / native_fps)

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames, actual_times, native_fps, frame_count


@torch.inference_mode()
def encode_frames(
    frames: Sequence[Image.Image],
    processor: AutoProcessor,
    model: AutoModel,
    batch_size: int,
    device: str,
    verbose: bool = False,
) -> torch.Tensor:
    chunks = []
    total_batches = math.ceil(len(frames) / batch_size)

    iterator = range(0, len(frames), batch_size)
    if verbose:
        iterator = tqdm(iterator, total=total_batches, desc="  Encoding batches", leave=False)

    for start in iterator:
        batch_frames = list(frames[start : start + batch_size])
        inputs = processor(images=batch_frames, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(**inputs)
        else:
            outputs = model(**inputs)

        feats = _extract_feature_tensor(outputs)

        chunks.append(feats.detach().cpu())

    return torch.cat(chunks, dim=0)


def save_feature_payload(
    out_path: Path,
    video_id: str,
    model_name: str,
    fps: float,
    features: torch.Tensor,
    timestamps: List[float],
    video_path: str,
    native_fps: float,
    frame_count: int,
) -> None:
    payload = {
        "video_id": video_id,
        "model_name": model_name,
        "sampling_fps": fps,
        "native_fps": native_fps,
        "frame_count": frame_count,
        "video_path": video_path,
        "timestamps": torch.tensor(timestamps, dtype=torch.float32),
        "features": features,
    }
    torch.save(payload, out_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)

    video_index = build_video_index(str(data_root), feature_root=str(feature_root), include_subtitles=False)
    if args.limit is not None:
        video_index = video_index[: args.limit]

    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=not args.use_slow_processor)
    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    manifest = []

    for sample in tqdm(video_index, desc="Caching SigLIP features"):
        out_path = feature_root / f"{sample.video_id}.pt"
        if out_path.exists() and not args.overwrite:
            manifest.append({
                "video_id": sample.video_id,
                "feature_path": str(out_path),
                "status": "skipped",
            })
            continue

        frames, timestamps, native_fps, frame_count = sample_frames_uniform(
            sample.video_path,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        if args.verbose:
            print(
                f"[cache] video_id={sample.video_id} sampled_frames={len(frames)} "
                f"native_fps={native_fps:.2f} frame_count={frame_count}"
            )
        features = encode_frames(
            frames=frames,
            processor=processor,
            model=model,
            batch_size=args.batch_size,
            device=device,
            verbose=args.verbose,
        )
        save_feature_payload(
            out_path=out_path,
            video_id=sample.video_id,
            model_name=args.model_name,
            fps=args.fps,
            features=features,
            timestamps=timestamps,
            video_path=sample.video_path,
            native_fps=native_fps,
            frame_count=frame_count,
        )
        manifest.append({
            "video_id": sample.video_id,
            "feature_path": str(out_path),
            "num_frames": int(features.shape[0]),
            "feature_dim": int(features.shape[-1]),
            "status": "cached",
        })

    manifest_path = feature_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
