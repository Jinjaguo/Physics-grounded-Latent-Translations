# EXP_F14：State-conditioned safe fallback control

## 来自 F13 的直接动机

F13 已经给出清楚结果：factor option graph 不优于 monolithic/scalar recovery；继续调整 option threshold 不构成新的科学推进。更关键的是，task05 instance28 中现有 `release → lift → base-policy replan` fallback 在 replan 开始前就显著恶化 world state。系统不能把“选择 fallback”与“fallback 是安全且可执行的”混为一谈。

因此 F14 将 fallback 从一个固定标签改为一组真实 control proposals，并检验 decision-time factors 是否能选择较少伤害、同时仍允许从当前真实 world state replan 的 fallback。

## 核心假设

1. fallback harm 具有可监督的阶段结构：一部分发生在 disengagement primitive，一部分发生在 base-policy continuation。
2. 部署时 visual support 与 gripper-holding proxy 可能决定是否应该 release、保持 gripper、直接从当前 state replan，或安全停止；没有独立监督时不声称直接观测 contact。
3. factor-conditioned fallback 至少应降低 immediate harmful intervention；它是否提高最终 task success 必须独立报告，不能用“安全停止”冒充 task recovery。

## 新数据

- 使用新的正式 simulator instances 32 和 33；instance 34 仅用于 smoke。
- 重新生成 confirmed-failure states，不复用 F13 outcome。
- cohort 同时包含 oracle-local 与 oracle-no-local states，避免把 severe 标签当作 no-local 真值。
- 每个 failure 保存 intervention 前的 visual/robot-state factors，以及各 fallback primitive 后和 continuation 后的反馈。

## 部署分支

在所有 outcome 发生前冻结下列 selected-only systems：

1. `factor_fallback`：若 support gate 接受 local，则执行同一 monolithic local recovery；否则由部署时 visual support 与 gripper-holding proxy 在 fallback proposals 中选择。
2. `release_lift_replan`：当前基线，先 open/release 并 lift，再从真实 current state 调 base policy。
3. `direct_current_state_replan`：不先执行 disengagement，直接从当前真实 state replan。
4. `closed_hold_replan`：短时保持 closed gripper，再从该真实 state replan。
5. `hold_only_safe_stop`：在固定评估 horizon 内保持/停止，不宣称 task completion。

禁止在执行全部 candidates 后回填 `factor_fallback`。factor rule 只能从 F9–F13 历史 training/diagnostic data 制定，并在 F14 formal cohort 前冻结。

## Factor 定义与控制语义

- `gripper holding proxy`：来自部署时 gripper proprioception，不使用 simulator object pose 或未来 outcome。
- `visual support`：来自部署时双相机与 robot state；它是 decision input，不是 local recoverability 真值。
- `local support`：沿用部署时 visual geometry/outcome gate，只决定 local-vs-fallback，不决定 oracle coverage。
- `safe-stop`：定义为固定 horizon 内没有超过预注册 harm boundary；它与 final task success 是两个不同指标。

预期 factor rule：闭爪时优先 `closed_hold_replan`；视觉上仍接近 support boundary 时优先 `direct_current_state_replan`；低 support 且没有可靠 holding evidence 时允许 `hold_only_safe_stop`。具体阈值必须在 formal cohort 前由历史数据冻结，且不能把该规则当作 recoverability 真值。

## Intervention 与 oracle protocol

1. 构造并验证 confirmed failure；保存可恢复 simulator state。
2. 仅用当前可部署观测计算 factors、support 和所有 deployment decisions。
3. 将全部 decisions 写盘并关闭任何 outcome 访问路径。
4. 对每个 system 从同一 pre-intervention state 执行 selected-only intervention。
5. 分别记录 fallback phase terminal state、base-policy continuation terminal state 和最终 feedback。
6. selected-only 全部完成后，才从同一 state 执行 local proposals 与全部 fallback proposals 的 oracle，用于 coverage 和 headroom。
7. 独立 audit decision-before-outcome、restore equality、step accounting、selected/oracle separation 与汇总重算。

## 必须报告的指标

- immediate fallback-phase harm rate；
- post-replan additional harm rate；
- final task success rate；
- safe-stop rate，且绝不并入 task success；
- oracle local coverage 与 oracle fallback-safe coverage；
- factor fallback selector regret；
- 按 visual-support/gripper-holding strata 的各 primitive outcome；
- proposal absent、selector miss、fallback primitive harm、continuation/handoff failure 的分解计数。

## 机制级比较

- factor selector vs 单一固定 fallback；
- release/lift vs direct replan：定位 disengagement 是否是 harm 来源；
- direct replan vs hold/replan：定位短时保持是否保护 held object；
- hold-only vs replan：区分安全性与任务恢复能力；
- 同一 local policy 的 supported states：确保差异确实来自 fallback，而非 local recovery 改动。

## 成功、否定与 pivot 条件

- 若 factor selector 降低 immediate harm 且不降低 final success，支持 factor 作为 fallback control interface。
- 若某个固定 primitive 同样好，factor conditioning 没有增益，应删除无价值 factor，不包装为正结果。
- 若 safe-stop 较安全但不成功，只支持 safety/fallback calibration claim，不支持 recovery claim。
- 若所有 fallback proposals 都伤害同一 state，则明确该 state 没有当前系统内的 safe local fallback；下一实验应引入可恢复的 task-level replan/safe restart mechanism，而不是继续调 selector。
- 若主要损失仍来自 local recovery，下一实验转向 proposal generation；若主要损失来自 continuation，则转向 handoff-aware replan。

## 预期 artifacts

- `experiments/EXP_F14/frozen_protocol.json`
- `experiments/EXP_F14/confirmed_failures.jsonl`
- `experiments/EXP_F14/factor_decisions.jsonl`
- `experiments/EXP_F14/selected_outcomes.jsonl`
- `experiments/EXP_F14/phase_feedback.jsonl`
- `experiments/EXP_F14/oracle_outcomes.jsonl`
- `experiments/EXP_F14/metrics.json`
- `experiments/EXP_F14/audit.json`
- `reports/EXP_F14_report.md`
- `reports/next_exp_fromF14.md`

F14 只有在产生新的 confirmed-failure data、实际 fallback interventions、phase feedback、measurable metrics 并通过 audit 后才占用 EXP ID。排障、依赖修复、smoke 和 interface gate 均不占用新的 EXP ID。
