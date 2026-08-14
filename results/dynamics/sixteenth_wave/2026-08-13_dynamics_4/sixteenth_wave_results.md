# PGLT 第十六轮 / 第四次动力学实验报告

## 结论

本轮从暂停边界继续执行，没有重做 Tier 0/Tier 1A。Tier 1B 对 **358 个 RoboVerse 文件、4,836 条轨迹**完成逐轨迹 pickle schema、动作维度和长度审计；最大长度 64，直接长候选为 0。Tier 2 staged 下载 `subset_training_023`，完成 SHA256、metadata/动作探针审计及清理。该 shard 携带全 ABCD training annotation 表（22,966 条、六任务各约 675 条），全表最大长度 **65 帧**，因此 `_000…_022` 只包含该表所索引的其他帧，继续下载不能产生 >=160 的直接 annotation。

指导文件要求 6 tasks × 10 段，因此 data-adequacy gate 为 **FAIL**。本轮没有读取任何 F1/F2 prospective metric、没有执行 bootstrap、没有拼接短轨迹，也没有读取未来动作。

C3c-local 保持 **SUPPORTED**。C3c-long 与 C3d 均为 **NOT_EVALUATED_INSUFFICIENT_DATA**，不是负结果。C3a/C3b 保持 **REJECTED**，不再进行 DEL rescue。

## 数据源审计

- `CollisionCode/calvin_d_d_lerobot_v2.1`: 12,614,629 bytes; SCOUTING_ONLY; direct >=160 candidates = 0.
- `RoboVerseOrg/roboverse_data`: 215,910,633 bytes; REJECTED; direct >=160 candidates = 0.
- `VyoJ/calvin-ABCD-D-subsets`: 2,160,142,328 bytes; PRIMARY_COMPATIBLE; direct >=160 candidates = 0.

本地 inventory 记录 **4728** 个 episode NPZ；三个既有 CALVIN 根合计见 `disk_budget.json`。LeRobot 始终是 10-Hz scouting-only；RoboVerse 不满足动作/时间兼容门；VyoJ 数据格式兼容但没有足够长的 annotation。

## 精确缺失数据

- `lift_blue_block_slider`: valid 0, missing 10.
- `lift_red_block_table`: valid 0, missing 10.
- `place_in_slider`: valid 0, missing 10.
- `push_pink_block_right`: valid 0, missing 10.
- `turn_off_lightbulb`: valid 0, missing 10.
- `turn_on_lightbulb`: valid 0, missing 10.

## Prospective collection 状态

- CALVIN commit：`fa03f01f19c65920e18cf37398a9ce859274af76`。
- 原生 pipeline：`third_party/calvin/calvin_env/calvin_env/vrdatacollector.py`。
- 控制/记录频率：30 Hz；最低 160 帧；最大 12 秒。
- 计划：只补上面列出的缺失 cell，保留成功与失败，不按 F1/F2 表现筛选。
- 当前 blocker：真实 VR/SHARED_MEMORY 与两项运行依赖缺失。
- 未采用范围外 workaround：没有新写随机、脚本或 learned behavior policy。

## 23 个明确回答

1. Existing local compatible roots: **3** roots (`data/representation/calvin_task_D_D`, `third_party/calvin/dataset/calvin_debug_dataset`, `archive/retired_snapshot/artifacts/third_wave/official_metadata/task_D_D`); none has a >=160-frame six-task annotation.
2. Public sources audited/downloaded: **CollisionCode LeRobot D/D metadata, RoboVerse CALVIN six-task files, and VyoJ original-format ABCD subset metadata/frames**.
3. Downloaded bytes: **12,614,629 / 215,910,633 / 2,160,142,328**, respectively.
4. Rejected/ineligible compatibility: **LeRobot is 10 Hz (SCOUTING_ONLY); RoboVerse is converted object/9-D joint-style data with unproven 30-Hz timing (REJECTED)**.
5. The 10-Hz LeRobot conversion remained scouting-only: **yes**; no interpolation or repetition was used.
6. Direct >=160-frame candidates: **0 LeRobot, 0 RoboVerse, 0 VyoJ ABCD**.
7. Valid segments per frozen task: **0 for every one of the six tasks**.
8. Open data reached 10/task and 60 total: **no (0/task, 0 total)**.
9. Missing collection: **10 segments for each of all six tasks; 60 total**, each >=160 contiguous frames in exact 30-Hz 7-D CALVIN rel_actions.
10. F1/F2 frozen before all prospective metrics: **yes**; hashes were recorded, update counts are zero, and no primary metric was read.
11. H1/H2/H4/H8 rollout starts: **0 / 0 / 0 / 0**.
12. Paired trajectory AUC with upper 95% CI below zero: **not tested; adequacy gate blocked bootstrap**.
13. F2 beat F1 at H4: **not tested**.
14. F2 beat F1 at H8: **not tested**.
15. F2 reduced H8 decoded-action error: **not tested**.
16. F2 reduced H8 execution off-manifold drift: **not tested**.
17. Refinement correction aligned with GT correction: **not tested**.
18. Empirical normal-to-manifold distance decreased: **not tested**.
19. C3c-long: **NOT_TESTED_INSUFFICIENT_DATA**, not rejected.
20. C3d: **NOT_TESTED**.
21. DEL remains a frozen negative baseline only: **yes; no retraining, retuning, or primary-bootstrap use**.
22. Defensible story: **semantic/executable coordinates and local refinement remain supported; stable long-horizon dynamics remains unresolved**.
23. Manual/VR collection remains necessary: **yes, only the 60 missing cells listed above**.

## 当前可辩护论文故事

**Language-grounded action coordinates are semantically addressable, executable, and locally predictable; refinement improves short-horizon prediction, but stable long-horizon latent dynamics remains unresolved.**

中文：**语言落地动作坐标具有语义可寻址性、可执行性与局部可预测性；refinement 改善短期预测，但稳定的长时域 latent 动力学仍未解决。**

## 下一实验

按 `targeted_missing_data_acquisition_plan.json` 只采集六任务各缺失的 10 段、每段至少 160 帧的 30-Hz 原始 CALVIN 7-D `rel_actions`；通过 60 段 gate 后，使用本轮已冻结的 F1/F2/semantic checkpoint 和 10,000 次 whole-trajectory bootstrap 完成 H1/H2/H4/H8 评估。不得适配模型或再次调 DEL。
