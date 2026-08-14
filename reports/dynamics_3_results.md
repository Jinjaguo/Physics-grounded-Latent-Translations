# PGLT 第十五轮 / 第三次动力学实验报告

## 结论

C3b（执行子空间 decoder-grounded variational dynamics）最终为 **REJECTED**；C3c（generic structured refinement）为 **SUPPORTED**。开发 hard gate=FAIL，official validation 的 F4 相对方向一致=False。冻结表示 optimizer/backward/EMA 步数均为 0，历史 full-latent DEL 负结论保持不变。

## 实验设计与执行完整性

沿用 wave-13 完全相同的 train/development/official-validation episode split 与非重叠 H=16、stride=16 窗口。共享 semantic predictor 参数量为 8336，standalone development MSE=0.453692。F2/F4 使用同一冻结 F1 初始化与相同 4 次迭代，均无未来 target action。F4 的 metric 为冻结 decoder 连续输出（含 gripper logit、不含阈值）的 JVP pullback，epsilon=1e-3，仅 potential 可训练。

## Development 主结果

| model | H1 exec MSE | H2 exec MSE | H1 decoded MSE | H2 exec kNN radius | normalized rollout AUC |
|---|---:|---:|---:|---:|---:|
| F1_execution_mlp | 0.768293 | 0.727379 | 0.026034 | 1.88728 | 0.713912 |
| F2_matched_refinement | 0.688568 | 0.613212 | 0.0235296 | 1.66664 | 0.621363 |
| F3_free_execution_del | 1.63566 | 2.0955 | 0.042346 | 3.7728 | 1.78095 |
| F4_decoder_geometry_del | 0.780406 | 0.74882 | 0.0262016 | 1.92715 | 0.729927 |

## One-shot official validation 主结果

| model | H1 exec MSE | H2 exec MSE | H1 decoded MSE | H2 exec kNN radius | normalized rollout AUC |
|---|---:|---:|---:|---:|---:|
| F1_execution_mlp | 0.892389 | 0.817622 | 0.0278869 | 1.95592 | 0.816219 |
| F2_matched_refinement | 0.802108 | 0.690218 | 0.02533 | 1.69195 | 0.712314 |
| F3_free_execution_del | 1.69976 | 1.97326 | 0.0462247 | 3.65152 | 1.7532 |
| F4_decoder_geometry_del | 0.899629 | 0.838433 | 0.028037 | 1.98402 | 0.829609 |

## Compatibility preflight 与 hard gate

F4 true/F1/F2/F4 residual mean 分别为 3659.2 / 2909.11 / 2910.47 / 2713.08。

- beats_F1_rollout_auc: **FAIL**
- beats_F2_rollout_auc: **FAIL**
- lowers_two_step_execution_off_manifold_drift: **FAIL**
- one_step_decoded_action_not_materially_worse: **PASS**
- residual_compatibility: **FAIL**
- residual_error_positive_alignment: **FAIL**
- solver_convergence_materially_above_historical_zero: **FAIL**
- solver_nonfinite_rate_zero: **PASS**

## 15 个明确回答

1. Restricting DEL to `z_exec` removes the residual mismatch: **no** (F3 true-next compatibility=False).
2. Decoder metric makes true-next lower-residual than F1: **no** (means 3659.2 vs 2909.11).
3. F4 residual is positively aligned with execution error: **no** (Spearman=-0.208636, bootstrap 95% [-0.258787, -0.15983]).
4. Execution-only free DEL outperforms historical full-latent DEL: **no** on development one-step execution MSE (1.63566 vs 1.6249).
5. F4 outperforms F1: **no** on development AUC (0.729927 vs 0.713912).
6. F4 outperforms matched F2: **no** (0.729927 vs 0.621363).
7. F4 reduces execution off-manifold drift: **no** (two-step kNN radius 1.92715 vs 1.88728).
8. F4 hybrid latent decodes more accurately than F1: **no** (one-step continuous MSE 0.0262016 vs 0.026034).
9. Semantic retention remains stable: **yes**; every execution model uses the identical shared semantic checkpoint and two-step retrieval range is 0.4–0.4.
10. F4 solver converges materially better than historical rate 0: **no** (development one-step convergence=0).
11. One-shot validation preserves the exact development ordering: **yes** (dev ['F2_matched_refinement', 'F1_execution_mlp', 'F4_decoder_geometry_del', 'F3_free_execution_del']; validation ['F2_matched_refinement', 'F1_execution_mlp', 'F4_decoder_geometry_del', 'F3_free_execution_del']).
12. C3b is **REJECTED**.
13. Generic structured refinement C3c is **SUPPORTED** (F2<F1 development=True, validation=True).
14. Defensible paper story: **Language defines meaningful action coordinates; structured refinement keeps learned transitions near executable regions.**
15. Next-wave longer trajectories: **yes**; carry ['F1_execution_mlp', 'F2_matched_refinement', 'historical_DEL_negative_baseline'].

## 科学上可辩护的故事与下一实验

Language defines meaningful action coordinates; structured refinement keeps learned transitions near executable regions.

中文：语言定义有意义且可执行的动作坐标；通用结构化 refinement 改善局部转移，而 DEL 仅保留为负机制基线。

下一实验：Expose longer annotation-consistent trajectories and make F1 execution MLP versus F2 matched generic refinement the sole primary comparison at horizons 1/2/4/8; retain DEL only as a frozen negative baseline and do not attempt another DEL rescue.

## 可复现性与存储

所有 exact commands、预注册、checkpoint、开发/验证 raw aggregate、residual/error、decoded/semantic/off-manifold 表、参数量、manifest、gate、claim JSON、pytest XML、环境和文件审计均保存在本 wave 目录。最终磁盘审计见 `final_integrity_check.json`。
