# DSCA-VL Reproduction Skeleton

This folder contains a runnable PyTorch skeleton for the DSCA-VL pipeline:

1. Frozen visual/text encoders
2. DSAM foreground-background decoupling
3. CGRM hierarchical causal graph reasoning
4. CMOS frame policy and GRPO utilities

## Structure

- `dscavl/encoders.py`: frozen visual encoder adapter + query text encoder stub
- `dscavl/dsam.py`: background prototypes, projection decoupling, semantic score and losses
- `dscavl/cgrm.py`: frame/event graph construction and hierarchical message passing
- `dscavl/policy.py`: per-frame Bernoulli policy head
- `dscavl/rewards.py`: accuracy/connectivity/sparsity rewards and frame-graph lifting
- `dscavl/grpo.py`: GRPO-style clipped objective with KL regularization
- `dscavl/model.py`: end-to-end DSCAVL model
- `train_stage1.py`: stage1 warmup loss (compact + orth + align)
- `train_stage2.py`: stage2 RL fine-tuning skeleton

## Quick Start

```bash
conda activate dsca_vl
pip install -r requirements.txt
python cache_siglip_features.py --data-root data --feature-root features --fps 1
python train_stage1.py
python train_stage2.py
python eval_proxy_mcq.py
```

If PyTorch installation fails on Windows, install `torch`, `torchvision`, and `torchaudio` separately with the wheel that matches your CUDA version, then run `pip install -r requirements.txt` again.

## Integration Notes

- Replace `FrozenVisualEncoder` with SigLIP extraction or precomputed frame features.
- Replace `QueryTextEncoder` with Qwen2-VL text embedding path.
- Replace random `reward_acc` in `train_stage2.py` with real answer generation + judge.
- For strict paper-style GRPO, sample `G=8` trajectories per sample and compute group-normalized advantages.
- `gt_mask` in stage1 must come from your chosen pseudo-label or timestamp supervision strategy.

## Data Layout

- `data/query/{video_id}.json`: list of question records for one video
- `data/video/{video_id}.mp4`: source video
- `data/subtitle/{video_id}.srt`: subtitle track
- `features/{video_id}.pt`: cached SigLIP frame features saved by `cache_siglip_features.py`

## Feature Cache Payload

Each cached feature file stores:

- `features`: frame embedding tensor with shape `[T, D]`
- `timestamps`: sampled frame timestamps in seconds
- `video_id`, `video_path`, `sampling_fps`, `native_fps`, `frame_count`, `model_name`

## Dataset API

`dscavl.data.build_video_index(...)` builds a video-level index for caching.

`dscavl.data.QuestionFeatureDataset(...)` flattens the per-video JSON files into question-level samples and loads `features/{video_id}.pt` as `precomputed_features` when available.

The current `train_stage1.py` and `train_stage2.py` read precomputed SigLIP features from `features/` and tokenize question text with the local `Qwen2-VL-2B-Instruct` tokenizer path when present.

`train_stage1.py` now builds weak `gt_mask` supervision from subtitle timing files by matching question/option keywords against subtitle segments.

`eval_proxy_mcq.py` evaluates the current model without MLLM by using selected-frame evidence and option-text similarity to compute proxy MCQ accuracy and frame-selection statistics.
