# DSCA-VL 项目架构与复现进度详细说明

## 1. 项目目标与当前定位

本项目目标是复现论文 DSCA-VL 的核心两阶段流程：

1. Stage1：表示学习（DSAM + CGRM）
2. Stage2：策略学习（CMOS + GRPO）

当前代码已经实现了可运行的端到端训练/评估骨架，并完成了数据索引、SigLIP 特征缓存、弱监督对齐、GRPO 组采样与代理评测闭环。

当前定位：
- 这是“工程可运行 + 方法主干完整”的复现版本。
- 还不是“论文最终实验同等设置”的完整版（主要缺 MLLM 闭环推理与论文级评测协议）。

---

## 2. 代码结构总览

### 2.1 顶层脚本

- `cache_siglip_features.py`：离线提取视频帧特征并缓存到 `features/{video_id}.pt`
- `train_stage1.py`：Stage1 训练（compact + orth + align）
- `train_stage2.py`：Stage2 训练（GRPO，当前为 G 轨迹组采样）
- `eval_proxy_mcq.py`：代理 MCQ 评估（不依赖 MLLM）

### 2.2 核心包 `dscavl/`

- `config.py`：统一超参数
- `data.py`：数据索引、样本结构、Dataset、collate
- `encoders.py`：视觉/文本编码器适配
- `dsam.py`：背景-前景解耦与三项损失
- `cgrm.py`：帧/事件图构建与层级消息传递
- `policy.py`：逐帧 Bernoulli 策略头
- `grpo.py`：GRPO 核心目标
- `rewards.py`：奖励分解（准确性、连通性、稀疏性）
- `proxy_mcq.py`：代理答案评分与奖励
- `weak_supervision.py`：字幕弱监督 `gt_mask` 构造
- `model.py`：总装模型（encoders + DSAM + CGRM + policy）
- `__init__.py`：模块导出接口

---

## 3. 模块与函数接口详解

以下按“文件 -> 函数/类 -> 功能 -> 输入/输出”说明。

## 3.1 `dscavl/config.py`

### `DSCAVLConfig` (dataclass)
功能：集中管理数据、模型、训练、强化学习、弱监督参数。

关键字段：
- 数据与编码：`data_root`, `feature_root`, `siglip_model_name`, `text_model_name_or_path`
- 维度与结构：`dim`, `bg_prototypes`, `temp_window`, `temp_stride`
- DSAM 损失系数：`lambda_compact`, `lambda_orth`, `lambda_align`
- GRPO 参数：`group_size`（当前 4）, `omega_cons`, `omega_sparse`, `beta_kl`
- 弱监督阈值：`weak_sup_min_hits`, `weak_sup_expand_sec`
- 训练：`lr`, `stage2_lr`, `stage1_epochs`, `stage2_epochs`

---

## 3.2 `dscavl/data.py`

### 数据结构

### `QuestionSample`
功能：问题级样本结构（问题文本、选项、答案、路径、字幕文本等）。

### `VideoSample`
功能：视频级样本结构（视频路径 + 问题列表）。

### 核心函数

### `_clean_subtitle_text(text: str) -> str`
功能：去掉 srt 序号/时间戳/html 标签，得到纯文本字幕。

### `load_subtitle_text(subtitle_path: Optional[str]) -> Optional[str]`
功能：读取并清洗字幕文本。

### `load_query_records(query_path: str) -> List[Dict[str, Any]]`
功能：读取单个 `query/{video_id}.json`。

### `_resolve_paths(root: Path, video_id: str, feature_dir: Optional[Path]) -> Dict[str, Optional[Path]]`
功能：解析视频、字幕、特征文件路径。

### `_build_question_samples(...) -> List[QuestionSample]`
功能：将单个视频的 question json 条目构造成问题级对象。

### `build_video_index(data_root: str, feature_root: Optional[str] = None, include_subtitles: bool = False) -> List[VideoSample]`
功能：构建视频级索引。
输入：`data_root` 下要求有 `query/video/subtitle`。
输出：按视频组织的样本列表。

### `flatten_questions(video_samples: List[VideoSample]) -> List[QuestionSample]`
功能：将视频级结构打平为问题级列表。

### `single_sample_collate(batch)`
功能：仅允许 `batch_size=1` 的严格 collate。

### `variable_feature_collate(batch)`
功能：支持可变帧长特征的 batch 组装，保留 `precomputed_features` 列表。

### `QuestionFeatureDataset(Dataset)`
- `__init__(...)`：构建索引并打平
- `__len__()`：样本数
- `_tokenize(text)`：问题文本 token 化
- `__getitem__(index)`：返回问题级样本，若特征存在则加载
  - `precomputed_features`: `[T, D]`
  - `timestamps`: `[T]`

---

## 3.3 `dscavl/encoders.py`

### `FrozenVisualEncoder(nn.Module)`
功能：视觉编码适配器。
- 若传入 `precomputed_features`，直接返回（用于离线 SigLIP 缓存）
- 否则走轻量 CNN stub（默认参数冻结）

接口：
- `forward(video_frames=None, precomputed_features=None) -> Tensor[B, T, D]`

### `QueryTextEncoder(nn.Module)`
功能：文本编码 stub（embedding + masked mean pooling + LayerNorm）。

接口：
- `forward(token_ids, attention_mask=None) -> Tensor[B, D]`

---

## 3.4 `dscavl/dsam.py`

### `DSAM(nn.Module)`
实现论文中前景-背景解耦主干。

#### `_compute_background_prototypes(f_raw)`
功能：用可学习查询 `q_bg` 对帧特征注意力池化，得到背景原型 `P_bg`。

#### `_project_to_bg(f_raw, p_bg)`
功能：将帧特征投影到背景子空间，得到 `F_bg`。

#### `_compactness_loss(f_raw, p_bg)`
功能：前景到背景原型的紧致性约束。
当前实现：先单位球归一化，再用余弦距离 `1 - cos`，数值尺度稳定。

#### `_orthogonality_loss(f_fg, f_bg)`
功能：正交约束。
当前实现：对归一化 `f_bg` 的 Gram 矩阵离对角项做二范数惩罚，提供更稳定梯度。

#### `_semantic_scores(f_fg, f_q)`
功能：计算查询相关语义分数 `S_sem`。

#### `_alignment_loss(s_sem, gt_mask)`
功能：弱监督正样本帧上的对齐损失（负对数似然）。

#### `forward(f_raw, f_q, gt_mask=None)`
输出字典：
- `F_fg`, `F_bg`, `P_bg`, `S_sem`
- `loss_compact`, `loss_orth`, `loss_align`

---

## 3.5 `dscavl/cgrm.py`

### `build_membership_matrix(t, window, stride, device)`
功能：构造帧到事件的 membership 矩阵 `M[T, M]`。

### `EventPooler(nn.Module)`
功能：滑窗池化得到事件节点表示，并注入事件索引位置编码。

### `CGRM(nn.Module)`
功能：层级因果图推理。

关键内部函数：
- `_temporal_adjacency(t, window, device)`：帧图邻接（高斯时距衰减）
- `_semantic_event_adjacency(e)`：事件语义相似图
- `_causal_event_adjacency(e)`：事件因果方向图（上三角掩码）
- `_one_step_message_passing(x, a, updater)`：单步消息传递

`forward(f_fg, s_sem, f_q)` 输出：
- `H_graph`：图增强帧特征 `[B,T,D]`
- `E`：事件特征 `[B,M,D]`
- `A_temp`：帧邻接 `[B,T,T]`
- `A_event`：事件邻接 `[B,M,M]`
- `S_graph`：图增强帧评分 `[B,T]`
- `membership`：帧事件关系 `[B,T,M]`

---

## 3.6 `dscavl/policy.py`

### `FramePolicyHead(nn.Module)`
功能：根据图特征和查询向量预测逐帧选择概率。

接口：
- `forward(h_graph, f_q, s_graph, sample=True, threshold=0.5)`
输出：
- `logits`, `probs`, `actions`

说明：
- `sample=True` 使用 Bernoulli 采样（训练）
- `sample=False` 使用阈值二值化（评估）
- 已含数值稳定处理：`nan_to_num + clamp`

---

## 3.7 `dscavl/grpo.py`

### `bernoulli_logprob(probs, actions, eps=1e-6)`
功能：轨迹动作对数概率。

### `compute_advantages(rewards, eps=1e-6)`
功能：组内标准化优势函数。
输入约定：`rewards` 形状 `[B, G]`。

### `grpo_clip_loss(logprob_new, logprob_old, adv, clip_eps=0.2)`
功能：PPO 风格 clipping 的策略损失。

### `kl_to_reference(logits, ref_logits)`
功能：当前策略与参考策略 Bernoulli KL。

### `grpo_objective(...)`
功能：返回 RL 训练项：
- `loss_rl = loss_pg + beta_kl * loss_kl`
- 同时返回 `loss_pg`, `loss_kl`, `logprob_new`, `logprob_old`

---

## 3.8 `dscavl/rewards.py`

### `accuracy_reward(predicted_answer, target_answer, judge_fn=None)`
功能：准确性奖励（默认严格匹配）。

### `connectivity_reward(a_frame, actions)`
功能：选中帧子图连通性密度奖励。

### `sparsity_reward(actions)`
功能：稀疏奖励（选帧比例越低越好）。

### `composite_reward(reward_acc, reward_cons, reward_sparse, omega_cons, omega_sparse)`
功能：总奖励组合。

### `build_frame_graph(a_temp, a_event, membership)`
功能：将事件图提升回帧图并归一化。

### `reward_bundle(...)`
功能：一次性返回 `R_acc/R_cons/R_sparse/R_total/A_frame`。

---

## 3.9 `dscavl/proxy_mcq.py`

### `normalize_option_text(option)`
功能：解析选项前缀标签（A/B/C/...）并提取文本。

### `extract_answer_label(answer)`
功能：从答案字段提取标准标签。

### `score_mcq_options(model, out, options, tokenizer, device)`
功能：
1. 根据 `actions` 聚合证据帧特征
2. 编码选项文本
3. 计算 evidence 与 option embedding 相似度得分

输出：`(labels, scores)`。

### `compute_mcq_exact_reward(...)`
功能：基于 `argmax(scores)` 与真实标签是否一致生成奖励。
输出：`(reward_tensor, pred_label, gt_label, scores)`。

---

## 3.10 `dscavl/weak_supervision.py`

### `SubtitleSegment` (dataclass)
字段：`start_sec/end_sec/text`。

### `parse_srt_segments(subtitle_path)`
功能：解析 srt 到片段列表。

### `extract_keywords(question, options)`
功能：从问题和选项抽关键词（含停用词过滤）。

### `match_subtitle_segments(segments, keywords, min_hits=1)`
功能：按关键词命中阈值筛字幕片段。

### `build_gt_mask_from_subtitles(subtitle_path, timestamps, question, options, min_hits=1, expand_sec=1.5)`
功能：根据字幕命中片段和时间膨胀构造帧级 bool mask。
输出：`Tensor[T]` 或 `None`。

---

## 3.11 `dscavl/model.py`

### `DSCAVL(nn.Module)`
聚合编码、DSAM、CGRM、策略头。

关键方法：
- `encode(video_frames, query_tokens, attention_mask=None, precomputed_features=None)`
- `_select_actions_infer(probs)`：推理阶段按比例阈值选帧
- `forward(..., mode="stage1"|"stage2"|"infer", gt_mask=None, ...)`

模式约定：
- `stage1`：返回编码 + DSAM + CGRM
- `stage2`：额外返回 policy 输出，`sample=True`
- `infer`：强制 top-k 规则选帧

---

## 3.12 `train_stage1.py`

### `stage1_loss(outputs, cfg)`
功能：`lambda_compact*compact + lambda_orth*orth + lambda_align*align`。

### `build_tokenizer(cfg, repo_root)`
功能：优先加载本地 tokenizer 路径，否则用模型名拉取。

### `build_dataloader(cfg, tokenizer, data_root, feature_root)`
功能：构建问题级训练集（含字幕）。

### `_build_stage1_gt_mask(batch, index, cfg, device)`
功能：从 subtitle + timestamp 构造弱监督 mask。

### `_compute_stage1_sample_loss(...)`
功能：对单个样本前向并汇总 loss 与弱监督统计。
输出含：
- `loss/loss_compact/loss_orth/loss_align`
- `gt_hit_ratio/gt_has_supervision`

### `run_stage1_epoch(...)`
功能：批级累积，返回 epoch 指标：
- `loss, loss_compact, loss_orth, loss_align`
- `gt_coverage, gt_hit_ratio`

### `main()`
功能：训练入口。

---

## 3.13 `train_stage2.py`

### `build_tokenizer(...)`、`build_dataloader(...)`
功能同 Stage1，Stage2 数据不使用字幕。

### `_compute_sample_stage2_loss(...)`
功能：单样本 G 轨迹组采样强化学习。
流程：
1. 对同一问题采样 `G=cfg.group_size` 条轨迹
2. 每条轨迹计算 `R_total`
3. 组内标准化得到 advantage
4. 用 `grpo_objective` 计算每条轨迹 RL loss
5. 轨迹平均得到样本 loss 与样本 reward

说明：
- 当前反传只使用 `rl_loss`（不混入 detached 正则项）

### `_backward_stage2_loss(...)`
功能：非有限值检查 + grad finite 检查 + clip + step。

### `_accumulate_stage2_batch(...)`
功能：对 batch 中每个样本累积 loss/reward 与异常计数。

### `run_stage2_epoch(...)`
功能：训练一个 epoch，输出诊断：
- `skip_samples_nonfinite`
- `skip_batches_nonfinite`
- `skip_grads_nonfinite`

### `main()`
功能：Stage2 入口。

---

## 3.14 `cache_siglip_features.py`

功能：视频帧离线缓存，供训练直接加载。

关键接口：
- `parse_args()`：含 `--fps/--batch-size/--max-frames/--overwrite/--limit/--verbose`
- `sample_frames_uniform(video_path, fps, max_frames)`：按秒均匀采样
- `encode_frames(frames, processor, model, batch_size, device, verbose=False)`
- `save_feature_payload(...)`
- `main()`

缓存文件 payload 字段：
- `features [T,D]`
- `timestamps [T]`
- `video_id, model_name, sampling_fps, native_fps, frame_count, video_path`

---

## 3.15 `eval_proxy_mcq.py`

功能：不依赖 MLLM 的代理评测。

核心流程：
1. 加载模型和数据
2. `mode="infer"` 得到选帧动作
3. `compute_mcq_exact_reward` 得到标签命中
4. 汇总输出 `mcq_accuracy/mean_selected_frames/mean_selected_ratio`

输出：
- 屏幕指标
- 每样本 jsonl 预测文件

---

## 4. 当前已完成工作（按里程碑）

## 4.1 工程与环境

- 已建立可运行 Python 项目结构
- 已打通 conda 环境 + torch/cuda 运行
- 已补齐依赖并可直接执行 Stage1/2/评估脚本

## 4.2 数据与特征

- 已完成数据索引逻辑（query/video/subtitle 对齐）
- 已完成问题级 Dataset + 可变长度特征 collate
- 已完成 SigLIP 离线缓存脚本
- 已完成全量特征缓存并生成 manifest

## 4.3 模型与训练

- DSAM/CGRM/Policy/GRPO/Reward 主干均已实现
- Stage1 已接入弱监督 `gt_mask`
- Stage2 已实现 G 轨迹组采样（当前 G=4）
- Stage2 已加入数值稳定机制（nan/inf 检查、梯度检查、梯度裁剪）

## 4.4 指标与评估

- 已有 Stage1 训练日志（loss 与弱监督统计）
- 已有 Stage2 训练日志（R_total 上升）
- 已有代理 MCQ 评估脚本与结果输出

---

## 5. 你当前这轮结果如何解读

你给出的最近训练日志（Stage1/Stage2）体现了三件事：

1. Stage1：`align` 连续下降，说明弱监督监督项在起作用
2. Stage1：`gt_coverage/gt_hit_ratio` 稳定，说明字幕匹配策略稳定
3. Stage2：`R_total` 持续上升且所有 skip 指标为 0，说明 RL 在稳定优化

整体判断：当前训练行为是健康的，已经脱离“脚本可跑但不学习”的阶段。

---

## 6. 与论文的差距评估

以下按“完成度等级”说明。

## 6.1 已基本对齐（约 70%~80%）

- 两阶段总体流程（Stage1 + Stage2）
- DSAM/CGRM/CMOS/GRPO 主干组件
- 组采样 GRPO（当前 G=4）
- 奖励分解思想（准确性+连通性+稀疏性）
- 弱监督时间对齐机制（字幕到帧 mask）

## 6.2 部分对齐（约 40%~60%）

- 文本编码：当前 `QueryTextEncoder` 仍是轻量 stub，不是完整 Qwen2-VL 表示路径
- 奖励准确性：当前用 proxy MCQ 相似度，不是论文级 MLLM 答题与 judge
- 训练细节：超参数搜索、warmup/cosine、多 seed 统计、消融配置尚未系统化

## 6.3 尚未对齐（约 0%~30%）

- 完整 MLLM 闭环（视频片段输入 -> 生成答案 -> 判分）
- 论文同款 benchmark protocol 与最终汇总表
- 大规模实验复现（跨任务、跨域、完整 ablation）

---

## 7. 粗粒度“离论文终稿还差多少”

给一个工程视角估计（不是论文官方定义）：

- 代码主干可运行度：约 85%
- 方法机制实现度：约 75%
- 论文实验对齐度：约 45%
- 论文最终可对标度：约 35%~50%

差距核心不是“模块缺失”，而是“评测闭环与实验协议未完全论文化”。

---

## 8. 建议的下一阶段任务（按收益排序）

1. 接入真实 MLLM 推理与判分链路（替换 proxy reward）
2. 固化论文评测 protocol（数据划分、指标口径、seed 统计）
3. 系统做超参数网格（`beta_kl`, `omega_*`, `group_size`, weak supervision thresholds）
4. 补全关键 ablation（去 DSAM / 去 CGRM / 去 R_cons / 去 R_sparse）
5. 产出与论文同格式结果表与误差分析

---

## 9. 文档结论

当前项目已经完成了从“论文思路”到“可训练系统”的关键跨越：
- 主干模块齐全
- 数据与特征链路闭合
- Stage1/Stage2 均可稳定训练
- 指标有可解释改善趋势

剩余工作主要集中在“论文级实验闭环”而非“核心代码从零重写”。换句话说，现在更像是进入了“实验工程与对标调参阶段”。
