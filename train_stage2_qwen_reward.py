"""Stage2 + Qwen 答题奖励（本文件内嵌简化版 ``QwenAnswerRewarder``，不依赖 qwen_reward_utils）。

与 ``train_stage2.py`` 类似但用真实 VLM 生成答案替代 proxy MCQ。
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Sequence

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from dscavl import (
    DSCAVL,
    DSCAVLConfig,
    QueryTextEncoder,
    QuestionFeatureDataset,
    variable_feature_collate,
    add_swanlab_train_args,
    resolve_swanlab_project,
    resolve_train_checkpoint_dir,
    swanlab_try_init,
)
from dscavl.grpo import compute_advantages, compute_stage2_grpo_terms, grpo_objective
from dscavl.proxy_mcq import extract_answer_label
from dscavl.rewards import reward_bundle
from dscavl.train_utils import add_cgrm_use_f_bg_cli_args, merge_cgrm_use_f_bg_cfg

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover
    process_vision_info = None

_PRED_LABEL_RE = re.compile(r"([A-Z])")


def parse_args() -> argparse.Namespace:
    """Qwen 路径、数据根、batch/group 等 CLI。"""
    parser = argparse.ArgumentParser(description="Train DSCA-VL stage2 with Qwen2-VL answer-based reward.")
    parser.add_argument("--data-root", type=str, default=None, help="Override cfg.data_root")
    parser.add_argument("--feature-root", type=str, default=None, help="Override cfg.feature_root")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory. Default: output/stage2_qwen_reward/<YYYYMMDD_HHMMSS> under repo root.",
    )
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override cfg.train_batch_size.")
    parser.add_argument("--group-size", type=int, default=None, help="Override cfg.group_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override cfg.num_workers.")
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default="/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct",
        help="Local path or HF id for Qwen2-VL model used in reward generation.",
    )
    parser.add_argument("--qwen-max-new-tokens", type=int, default=32, help="Max generated tokens for reward answer.")
    parser.add_argument("--max-selected-frames", type=int, default=8, help="Max selected frames passed to Qwen.")
    parser.add_argument(
        "--eval-mcq-only",
        action="store_true",
        help="Use only MCQ samples (options non-empty) for reward/training signal.",
    )
    parser.add_argument(
        "--sanity-steps",
        type=int,
        default=0,
        help="Run only N optimizer steps per epoch for quick sanity validation (<=0 means full epoch).",
    )
    add_swanlab_train_args(parser)
    add_cgrm_use_f_bg_cli_args(parser)
    return parser.parse_args()


def build_tokenizer(cfg: DSCAVLConfig, repo_root: Path):
    """加载 DSCA 侧 query 用 tokenizer。"""
    tokenizer_path = repo_root / cfg.text_model_name_or_path
    source = str(tokenizer_path) if tokenizer_path.exists() else cfg.text_model_name_or_path
    if source == "Qwen2-VL-2B-Instruct":
        local_qwen = Path("/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct")
        if local_qwen.exists():
            source = str(local_qwen)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloader(cfg: DSCAVLConfig, tokenizer, data_root: Path, feature_root: Path) -> DataLoader:
    """构建带预计算特征的 shuffle DataLoader。"""
    dataset = QuestionFeatureDataset(
        data_root=str(data_root),
        feature_root=str(feature_root),
        tokenizer=tokenizer,
        max_length=cfg.query_max_length,
        include_subtitles=False,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=variable_feature_collate,
    )


class QwenAnswerRewarder:
    """内联 Qwen2-VL：按选中时间戳单帧解码（无 LRU），生成 MCQ 答案并解析标签。"""

    def __init__(self, model_path: str, max_new_tokens: int = 32, max_selected_frames: int = 8):
        """bf16 GPU / fp32 CPU 加载生成模型并 eval。"""
        self.max_new_tokens = max_new_tokens
        self.max_selected_frames = max_selected_frames
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _decode_frame_at_time(video_path: str, ts_sec: float) -> Image.Image | None:
        """OpenCV 按时间 seek 一帧。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.0:
            fps = 30.0
        frame_index = min(max(int(round(ts_sec * fps)), 0), max(frame_count - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def _select_indices(self, actions: torch.Tensor, probs: torch.Tensor) -> list[int]:
        """与共享工具相同：动作非零或 argmax，再按 prob top-k 截断。"""
        # actions/probs shape: [T]
        idx = torch.nonzero(actions > 0.5, as_tuple=False).squeeze(-1).tolist()
        if not idx:
            idx = [int(torch.argmax(probs).item())]
        if len(idx) > self.max_selected_frames:
            # keep top-prob selected frames when too many are selected
            prob_vals = probs[idx]
            topk = torch.topk(prob_vals, k=self.max_selected_frames).indices.tolist()
            idx = [idx[i] for i in topk]
            idx.sort()
        return idx

    @staticmethod
    def _build_mcq_prompt(question: str, options: Sequence[str]) -> str:
        """构造多选项题干。"""
        if options:
            options_text = "\n".join(options)
            return (
                f"{question}\n"
                f"Options:\n{options_text}\n"
                "Please answer with only the option letter (A/B/C/...)."
            )
        return question

    @torch.no_grad()
    def generate_answer(
        self,
        video_path: str,
        timestamps: torch.Tensor | None,
        actions: torch.Tensor,
        probs: torch.Tensor,
        question: str,
        options: Sequence[str],
    ) -> str:
        """解码选中帧 → 多图 chat → ``generate`` → 去特殊符号的文本。"""
        if timestamps is None or timestamps.numel() == 0:
            return ""

        selected_idx = self._select_indices(actions, probs)
        images: list[Image.Image] = []
        times = timestamps.float().cpu().tolist()
        for i in selected_idx:
            if 0 <= i < len(times):
                frame = self._decode_frame_at_time(video_path, float(times[i]))
                if frame is not None:
                    images.append(frame)
        if not images:
            return ""

        prompt = self._build_mcq_prompt(question, options)
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out_ids)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer.strip()

    @staticmethod
    def mcq_reward(pred_text: str, gt_answer: str, device: torch.device) -> tuple[torch.Tensor, str, str]:
        """正则抽预测字母并与 ``extract_answer_label(gt)`` 比较。"""
        pred_match = _PRED_LABEL_RE.search(pred_text.upper())
        pred_label = pred_match.group(1) if pred_match else ""
        gt_label = extract_answer_label(gt_answer)
        reward = 1.0 if pred_label and pred_label == gt_label else 0.0
        return torch.tensor([reward], device=device), pred_label, gt_label


def _compute_sample_stage2_loss(
    model: DSCAVL,
    old_policy,
    ref_policy,
    cfg: DSCAVLConfig,
    rewarder: QwenAnswerRewarder,
    query: torch.Tensor,
    query_mask: torch.Tensor | None,
    precomputed_features: torch.Tensor,
    video_path: str,
    timestamps: torch.Tensor | None,
    question: str,
    options: list[str],
    answer: str,
    device: torch.device,
):
    """单样本 Qwen 闭环 GRPO：多轨迹、``reward_bundle``、优势与 KL。"""
    precomputed_features = F.layer_norm(precomputed_features, (precomputed_features.shape[-1],))

    rollout_outputs = []
    rollout_rewards = []
    rollout_acc = []
    group_size = max(int(cfg.group_size), 1)

    for _ in range(group_size):
        backbone = model(
            None,
            query,
            mode="stage2_backbone",
            attention_mask=query_mask,
            precomputed_features=precomputed_features,
        )
        policy_kw = {
            "h_event": backbone.get("H_event"),
            "s_event": backbone.get("S_event"),
            "membership": backbone.get("membership"),
            "frame_aux": backbone.get("frame_patch_stats"),
            "event_aux": backbone.get("event_patch_stats"),
        }
        with torch.no_grad():
            old_out = old_policy(
                backbone["H_graph"],
                backbone["f_q"],
                backbone["S_graph"],
                sample=True,
                intra_event_topk=False,
                **policy_kw,
            )

        rollout_out = {
            **backbone,
            "actions": old_out["actions"],
            "probs": old_out["probs"],
            "probs_old": old_out["probs"],
            "event_actions": old_out.get("event_actions"),
        }

        pred_text = rewarder.generate_answer(
            video_path=video_path,
            timestamps=timestamps,
            actions=rollout_out["actions"][0].detach().cpu(),
            probs=rollout_out["probs"][0].detach().cpu(),
            question=question,
            options=options,
        )
        reward_acc, _, _ = rewarder.mcq_reward(pred_text, answer, device=device)
        rb = reward_bundle(
            reward_acc=reward_acc,
            a_temp=rollout_out["A_temp"],
            a_event=rollout_out["A_event"],
            membership=rollout_out["membership"],
            actions=rollout_out["actions"],
            omega_cons=cfg.omega_cons,
            omega_sparse=cfg.omega_sparse,
            omega_coverage=getattr(cfg, "omega_coverage", 0.2),
            frame_patch_stats=rollout_out.get("frame_patch_stats"),
        )

        reward_scalar = torch.nan_to_num(rb["R_total"].mean().detach(), nan=0.0, posinf=0.0, neginf=0.0)
        rollout_outputs.append(rollout_out)
        rollout_rewards.append(reward_scalar)
        rollout_acc.append(float(reward_acc.item()))

    if not rollout_outputs:
        zero = torch.zeros((), device=device)
        return zero, 0.0, 0.0

    rewards = torch.stack(rollout_rewards).to(device)
    advantages = compute_advantages(rewards.unsqueeze(0)).squeeze(0)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)

    total_loss = torch.zeros((), device=device)
    valid_rollouts = 0

    for ridx, out in enumerate(rollout_outputs):
        lp_new, lp_old, kl_pairs = compute_stage2_grpo_terms(
            model,
            old_policy,
            ref_policy,
            out,
            out["actions"],
            out.get("event_actions"),
        )
        rl = grpo_objective(
            lp_new,
            lp_old,
            advantages[ridx : ridx + 1],
            cfg.beta_kl,
            kl_logits_pairs=kl_pairs,
        )
        dsam_reg = cfg.stage2_dsam_reg_scale * (
            0.5 * out["loss_orth"]
            + 0.5 * out["loss_recon"]
        )
        rollout_loss = rl["loss_rl"] + dsam_reg
        if not torch.isfinite(rollout_loss):
            continue
        total_loss = total_loss + rollout_loss
        valid_rollouts += 1

    if valid_rollouts == 0:
        zero = torch.zeros((), device=device)
        return zero, rewards.mean().detach().item(), sum(rollout_acc) / max(len(rollout_acc), 1)

    sample_loss = total_loss / valid_rollouts
    sample_reward = rewards.mean().detach().item()
    sample_acc = sum(rollout_acc) / max(len(rollout_acc), 1)
    return sample_loss, sample_reward, sample_acc


def _extract_batch_tensors(batch, device: torch.device):
    """batch 内 query token 搬到 device。"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _backward_stage2_loss(model: DSCAVL, optimizer: AdamW, loss: torch.Tensor, cfg: DSCAVLConfig) -> bool:
    """反传、逐参数 grad 有限性、裁剪、step。"""
    if not torch.isfinite(loss):
        return False

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_finite = True
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            grad_finite = False
            break
    if not grad_finite:
        optimizer.zero_grad(set_to_none=True)
        return False

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
    optimizer.step()
    return True


def run_stage2_epoch(
    model: DSCAVL,
    ref_policy,
    rewarder: QwenAnswerRewarder,
    optimizer: AdamW,
    dataloader: DataLoader,
    cfg: DSCAVLConfig,
    device: torch.device,
    eval_mcq_only: bool = False,
    epoch_id: int = 1,
    sanity_steps: int = 0,
) -> tuple[float, float, float]:
    """epoch 内按 batch 聚合样本 loss 后反传；统计 loss/reward/acc 均值。"""
    total_loss = 0.0
    total_reward = 0.0
    total_acc = 0.0
    step_count = 0
    skipped_nonfinite_samples = 0
    skipped_empty_batches = 0
    skipped_backward = 0

    progress = tqdm(dataloader, desc=f"[Stage2-Qwen] Epoch {epoch_id}", leave=False)
    for batch in progress:
        features_list = batch.get("precomputed_features")
        if features_list is None:
            continue

        input_ids, attention_mask = _extract_batch_tensors(batch, device)
        options_batch = batch.get("options")
        answers_batch = batch.get("answer")
        questions = batch.get("question")
        video_paths = batch.get("video_path")
        timestamps_list = batch.get("timestamps")
        old_policy = copy.deepcopy(model.policy).eval()

        batch_loss = torch.zeros((), device=device)
        batch_reward = 0.0
        batch_acc = 0.0
        valid_count = 0

        for i, features in enumerate(features_list):
            if features is None:
                continue
            if eval_mcq_only and not options_batch[i]:
                continue

            precomputed_features = features.unsqueeze(0).to(device)
            query = input_ids[i : i + 1]
            query_mask = attention_mask[i : i + 1] if attention_mask is not None else None
            timestamps = timestamps_list[i]
            if isinstance(timestamps, torch.Tensor):
                timestamps = timestamps.cpu()

            sample_loss, sample_reward, sample_acc = _compute_sample_stage2_loss(
                model=model,
                old_policy=old_policy,
                ref_policy=ref_policy,
                cfg=cfg,
                rewarder=rewarder,
                query=query,
                query_mask=query_mask,
                precomputed_features=precomputed_features,
                video_path=video_paths[i],
                timestamps=timestamps,
                question=questions[i],
                options=options_batch[i],
                answer=answers_batch[i],
                device=device,
            )
            if not torch.isfinite(sample_loss):
                skipped_nonfinite_samples += 1
                continue
            batch_loss = batch_loss + sample_loss
            batch_reward += sample_reward
            batch_acc += sample_acc
            valid_count += 1

        if valid_count == 0:
            skipped_empty_batches += 1
            continue

        loss = batch_loss / valid_count
        if not _backward_stage2_loss(model, optimizer, loss, cfg):
            skipped_backward += 1
            continue

        total_loss += loss.detach().item()
        total_reward += batch_reward / valid_count
        total_acc += batch_acc / valid_count
        step_count += 1
        progress.set_postfix(
            loss=f"{(total_loss / max(step_count, 1)):.4f}",
            reward=f"{(total_reward / max(step_count, 1)):.4f}",
            acc=f"{(total_acc / max(step_count, 1)):.4f}",
        )
        if sanity_steps > 0 and step_count >= sanity_steps:
            break

    print(
        "[Stage2-Qwen][diag] "
        f"steps={step_count}, "
        f"skip_nonfinite_samples={skipped_nonfinite_samples}, "
        f"skip_empty_batches={skipped_empty_batches}, "
        f"skip_backward={skipped_backward}"
    )

    mean_loss = total_loss / max(step_count, 1)
    mean_reward = total_reward / max(step_count, 1)
    mean_acc = total_acc / max(step_count, 1)
    return mean_loss, mean_reward, mean_acc


def _save_stage2_checkpoint(
    ckpt_dir: Path,
    model: DSCAVL,
    optimizer: AdamW,
    cfg: DSCAVLConfig,
    epoch: int,
    mean_loss: float,
    mean_reward: float,
    mean_acc: float,
    is_best: bool,
):
    """保存整模 + policy 子字典及 MCQ 准确率指标。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"loss": mean_loss, "reward": mean_reward, "mcq_acc": mean_acc}
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "policy_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "metrics": metrics,
    }
    epoch_path = ckpt_dir / f"checkpoint-epoch{epoch}.pt"
    latest_path = ckpt_dir / "checkpoint-latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    if is_best:
        torch.save(payload, ckpt_dir / "checkpoint-best.pt")


def main():
    """初始化 Qwen 奖励器、仅训 policy、多 epoch 训练与 SwanLab。"""
    args = parse_args()
    cfg = DSCAVLConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.feature_root:
        cfg.feature_root = args.feature_root
    if args.train_batch_size is not None:
        cfg.train_batch_size = max(int(args.train_batch_size), 1)
    if args.group_size is not None:
        cfg.group_size = max(int(args.group_size), 1)
    if args.num_workers is not None:
        cfg.num_workers = max(int(args.num_workers), 0)
    merge_cgrm_use_f_bg_cfg(cfg, None, args.cgrm_use_f_bg, args.no_cgrm_f_bg)

    repo_root = Path(__file__).resolve().parent
    data_root = Path(cfg.data_root) if Path(cfg.data_root).is_absolute() else repo_root / cfg.data_root
    feature_root = Path(cfg.feature_root) if Path(cfg.feature_root).is_absolute() else repo_root / cfg.feature_root
    print(
        "[Stage2-Qwen] runtime cfg: "
        f"batch_size={cfg.train_batch_size}, group_size={cfg.group_size}, num_workers={cfg.num_workers}, "
        f"cgrm_use_f_bg={cfg.cgrm_use_f_bg}"
    )

    tokenizer = build_tokenizer(cfg, repo_root)
    dataloader = build_dataloader(cfg, tokenizer, data_root, feature_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = QueryTextEncoder(vocab_size=len(tokenizer), dim=cfg.dim)
    model = DSCAVL(cfg, text_encoder=text_encoder).to(device)
    ref_policy = copy.deepcopy(model.policy).eval()

    # Stage2: update policy only for stability.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.policy.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    ckpt_dir = resolve_train_checkpoint_dir(repo_root, args.checkpoint_dir, "stage2_qwen_reward")
    print(f"[Stage2-Qwen] checkpoint_dir={ckpt_dir}")

    rewarder = QwenAnswerRewarder(
        model_path=args.qwen_model_path,
        max_new_tokens=args.qwen_max_new_tokens,
        max_selected_frames=args.max_selected_frames,
    )

    project = resolve_swanlab_project(args, cfg)
    swan_ctx = swanlab_try_init(
        disabled=args.disable_swanlab,
        project=project,
        experiment_name=args.swanlab_experiment_name,
        experiment_name_prefix="stage2-qwen-reward",
        workspace=args.swanlab_workspace,
        log_prefix="Stage2-Qwen",
        config={
            "stage": "stage2_qwen_reward",
            "swanlab_project": project,
            "data_root": str(data_root),
            "feature_root": str(feature_root),
            "cgrm_use_f_bg": cfg.cgrm_use_f_bg,
            "epochs": cfg.stage2_epochs,
            "batch_size": cfg.train_batch_size,
            "lr": cfg.stage2_lr,
            "weight_decay": cfg.weight_decay,
            "text_model_name_or_path": cfg.text_model_name_or_path,
            "qwen_model_path": args.qwen_model_path,
            "group_size": cfg.group_size,
            "beta_kl": cfg.beta_kl,
            "omega_cons": cfg.omega_cons,
            "omega_sparse": cfg.omega_sparse,
            "qwen_max_new_tokens": args.qwen_max_new_tokens,
            "max_selected_frames": args.max_selected_frames,
            "eval_mcq_only": args.eval_mcq_only,
            "checkpoint_dir": str(ckpt_dir),
        },
    )

    best_reward = float("-inf")
    try:
        for epoch in range(cfg.stage2_epochs):
            epoch_id = epoch + 1
            mean_loss, mean_reward, mean_acc = run_stage2_epoch(
                model=model,
                ref_policy=ref_policy,
                rewarder=rewarder,
                optimizer=optimizer,
                dataloader=dataloader,
                cfg=cfg,
                device=device,
                eval_mcq_only=args.eval_mcq_only,
                epoch_id=epoch_id,
                sanity_steps=args.sanity_steps,
            )
            print(
                f"[Stage2-Qwen] epoch={epoch_id}, "
                f"loss={mean_loss:.4f}, "
                f"R_total={mean_reward:.4f}, "
                f"R_acc(mcq)={mean_acc:.4f}"
            )

            swan_ctx.log(
                {
                    "stage2_qwen/epoch": epoch_id,
                    "stage2_qwen/loss": mean_loss,
                    "stage2_qwen/R_total": mean_reward,
                    "stage2_qwen/R_acc_mcq": mean_acc,
                },
                log_prefix="Stage2-Qwen",
            )

            is_best = mean_reward > best_reward
            if is_best:
                best_reward = mean_reward
            _save_stage2_checkpoint(
                ckpt_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch_id,
                mean_loss=mean_loss,
                mean_reward=mean_reward,
                mean_acc=mean_acc,
                is_best=is_best,
            )
    finally:
        swan_ctx.finish(log_prefix="Stage2-Qwen")


if __name__ == "__main__":
    main()
