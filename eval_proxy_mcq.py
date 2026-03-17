from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dscavl import DSCAVL, DSCAVLConfig, QueryTextEncoder, QuestionFeatureDataset, compute_mcq_exact_reward, variable_feature_collate
from train_stage2 import build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSCA-VL with proxy MCQ scoring.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint path.")
    parser.add_argument("--output", type=str, default="eval_proxy_mcq.jsonl", help="Where to write per-sample predictions.")
    parser.add_argument("--batch-size", type=int, default=1, help="Eval dataloader batch size.")
    return parser.parse_args()


def load_model(cfg: DSCAVLConfig, tokenizer, checkpoint: str | None, device: torch.device) -> DSCAVL:
    model = DSCAVL(cfg, text_encoder=QueryTextEncoder(vocab_size=len(tokenizer), dim=cfg.dim)).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = DSCAVLConfig()
    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / cfg.data_root
    feature_root = repo_root / cfg.feature_root

    tokenizer = build_tokenizer(cfg, repo_root)
    dataset = QuestionFeatureDataset(
        data_root=str(data_root),
        feature_root=str(feature_root),
        tokenizer=tokenizer,
        max_length=cfg.query_max_length,
        include_subtitles=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=variable_feature_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, tokenizer, args.checkpoint, device)

    total = 0
    correct = 0
    selected_frames = 0.0
    selected_ratio = 0.0
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features_list = batch["precomputed_features"]

            for i, features in enumerate(features_list):
                if features is None:
                    continue

                out = model(
                    None,
                    input_ids[i : i + 1],
                    mode="infer",
                    attention_mask=attention_mask[i : i + 1],
                    precomputed_features=features.unsqueeze(0).to(device),
                )

                reward_acc, pred_label, gt_label, scores = compute_mcq_exact_reward(
                    model=model,
                    out=out,
                    options=batch["options"][i],
                    answer=batch["answer"][i],
                    tokenizer=tokenizer,
                    device=device,
                )

                actions = out["actions"][0]
                num_selected = float(actions.sum().item())
                num_frames = float(actions.numel())

                total += 1
                correct += int(reward_acc.item())
                selected_frames += num_selected
                selected_ratio += num_selected / max(num_frames, 1.0)

                rows.append(
                    {
                        "video_id": batch["video_id"][i],
                        "question_id": batch["question_id"][i],
                        "pred_label": pred_label,
                        "gt_label": gt_label,
                        "correct": int(reward_acc.item()),
                        "num_selected_frames": num_selected,
                        "num_total_frames": num_frames,
                        "selected_ratio": num_selected / max(num_frames, 1.0),
                        "scores": scores.detach().cpu().tolist(),
                    }
                )

    output_path = repo_root / args.output
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc = correct / max(total, 1)
    mean_selected_frames = selected_frames / max(total, 1)
    mean_selected_ratio = selected_ratio / max(total, 1)

    print(f"samples={total}")
    print(f"mcq_accuracy={acc:.4f}")
    print(f"mean_selected_frames={mean_selected_frames:.2f}")
    print(f"mean_selected_ratio={mean_selected_ratio:.4f}")
    print(f"saved_predictions={output_path}")


if __name__ == "__main__":
    main()