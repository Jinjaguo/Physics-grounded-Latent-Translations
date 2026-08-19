# Next Experiment from EXP_F1：EXP_F2 Task-Milestone Feedback Proposals

## F1 改变了什么

F1 把 failure detection 从问题中移除并建立了 6 个可精确恢复的 confirmed-failure checkpoints。42 条实际 intervention 表明，当前 proposal union 只覆盖 2/6；native 只覆盖 1/6；drawer 与 microwave 的 4 个状态在所有 local proposals 和“open-retreat + full-task replan”fallback 下都失败。与此同时，12 个成功 outcome 全部依赖 proposal 后 101–141 个真实 handoff steps，证明 consequence label 必须覆盖 deployed continuation。

因此主线 Priority A 的最小接口已经成立，但 Priority B 尚未解决。F2 不得训练 selector，也不应对同一个 5-step chunk 调 threshold、candidate 数或 learning rate。

## 应放弃的 formulation

以下 formulation 在 F1 的多阶段 articulated tasks 上应停止作为主要 proposal：

- 只执行一个固定 5-step open-loop chunk，然后把所有剩余结构交回 full-task prompt；
- 把“安全垂直退开”本身当成足够的 fallback；
- 假设一次 nearest action-chunk retrieval 能恢复 drawer/microwave 的阶段与接触关系；
- 用 task05 的高 coverage 推断跨任务 headroom。

这些组件可以保留为 matched baselines，但不能成为 F2 的新方法。

## 支配性瓶颈与竞争解释

F1 的 4 个 articulated-task failures 可能来自：

1. proposal duration 太短，尚未达到一个 base policy 可识别的 familiar milestone；
2. full-task prompt 无法从扰动状态推断当前未完成 subtask，handoff prompt/phase 错；
3. raw retrieval 没有根据当前 object displacement 适配 action；
4. 160-step continuation 只是不够长，而 proposal 本身并非问题；
5. 当前状态在本地不可恢复，必须更强 replan 或最终 environment-level fallback。

F2 必须在同一个实验中区分这些解释。

## 新科学假设

> 对需要“放入 articulated receptacle 并关闭”的 confirmed failures，proposal 应该是一个短时闭环、可重观察的 task milestone controller，而不是 5-step open-loop action chunk。显式把任务分解为 object acquisition/placement 与 articulation closure，并从真实 feedback 切换 milestone，会增加 oracle coverage；若只延长 full-task handoff 仍失败，则收益来自 proposal structure 而不是额外控制预算。

这是新的 proposal/control formulation，直接推进 Priority B；它不重新引入 failure detection。

## F2 将实际实现的 materially different directions

### 1. Closed-loop semantic milestone proposal

为每个实际 task 从 BDDL/instruction 固定两个可执行 milestone prompts：

- task03：`put the black bowl in the bottom drawer` → `close the bottom drawer`；
- task05：`pick up/place the book in the caddy`，作为已可恢复 reference；
- task09：`put the yellow and white mug in the microwave` → `close the microwave`。

每个 proposal 不是预生成 5-step chunk，而是在 30–60 个控制步内每 5 步从真实 observation 重新调用 frozen π0.5。proposal 保存每次 request、action、post-state 和 official predicate。milestone 到时不使用未来 outcome 选择 proposal；F2 oracle matrix 仍只用于 evaluator coverage。

### 2. State-adapted retrieval feedback proposal

从 train successful trajectories 检索相同 task 的近邻 issue point，但不直接 replay raw chunk。用当前 EEF/object 与 retrieved EEF/object 的差异对前若干 translation actions做明确的 task-space offset adaptation；随后每 5 步重观察并重新检索。这检验 F1 retrieval 的问题是 open-loop state mismatch 还是整个 retrieval family 无效。

### 3. Task-structured fallback/replan

fallback 仍不 reset world。它先 open-gripper disengage，然后根据当前未满足的 task milestone调用对应 subtask prompt；完成 placement 后再调用 closure prompt，最后返回 full task/base policy。它与 local milestone proposal的区别是：fallback 不要求短局部恢复后直接 handoff，而是明确放弃 local chunk，执行完整 safe replan path。

### 4. Equal-budget control

对同一 failure state运行一个与新 structured method拥有相同总控制预算、但始终使用原 full-task prompt 的 `extended_full_task_replan`。这样可以判定 coverage gain 是 task structure 还是简单延长 horizon。F1 的 5-step families与原 fallback继续作为历史 matched baselines。

## 数据与 intervention

- 首先在 F1 的 6 个 snapshots 上做 matched same-state comparison，直接测新增 family 的 marginal coverage；
- 再生成至少 task03/task09 各 2 个新 confirmed-failure executions（不同初始 object offsets / disturbance directions），避免结论只依赖 F1 的四个失败状态；
- 每个新 choice 都必须实际执行 proposal → reobserve → exact handoff/fallback → official metric；
- 新运行保存独立 `experiments/EXP_F2/` artifacts，不修改 F1。

F2 完成至少需要：新 failure data 或新 executed proposal outcomes、而不是对 F1 JSON 的离线重标。

## baselines 与机制诊断

- F1 native 5-step + fresh full-task handoff；
- F1 safe-disengage + full-task fallback；
- equal-budget extended full-task replan；
- closed-loop semantic milestone proposal；
- state-adapted retrieval feedback proposal；
- task-structured fallback。

必须分别报告：proposal-immediate milestone progress、exact handoff final success、family marginal coverage、fallback-only success、task/instance heterogeneity、执行步数、invalid/unsafe predicate、以及“placement 成功但 closure 失败”等 task-specific阶段失败。不要只报总 success。

## 支持与否定模式

支持假设：

- task03/task09 union oracle coverage 相比 F1 0/4 有实际增加；
- structured milestone proposal 或 structured fallback在同预算下严格优于 extended full-task prompt；
- 成功链的 saved trace 显示 placement/closure milestone 按真实 observation推进；
- 新 family 对 union coverage 有正 leave-family-out marginal contribution。

否定假设：

- 所有 structured choices仍为 0，或只与延长 full-task budget同步改善；
- task-specific milestone traces显示未产生不同状态/动作；
- 所有新增成功都可由 equal-budget full prompt复现；
- 新 states 与 F1 同样完全无 proposal headroom。

若否定，F3 不得换几个 duration/seed再跑；应转向 task-space planning、learned local feedback recovery policy或更明确的 controller-level manipulation skill。

## 完成 F2 前必须存在的 machine-verifiable artifacts

```text
experiments/EXP_F2/frozen_protocol.json
experiments/EXP_F2/confirmed_failures.jsonl
experiments/EXP_F2/failures/ (若生成新 states)
experiments/EXP_F2/proposal_cycles.jsonl
experiments/EXP_F2/rollouts/*.npz
experiments/EXP_F2/outcomes.jsonl
experiments/EXP_F2/coverage_matrix.csv
experiments/EXP_F2/metrics.json
experiments/EXP_F2/run_metadata.json
experiments/EXP_F2/audit.json
reports/EXP_F2_report.md
reports/next_exp_fromF2.md
```

audit 必须从逐步 traces 重算 outcome、milestone sequence、coverage 与 budget matching。只有真实新增 intervention 与审计通过后，F2 才能占用编号。

## 对最终研究主线的推进

F2 只解决 proposal generator/fallback coverage，并保持其与 consequence predictor 分离。只有 articulated-task headroom 实质提高后，F3 才能冻结 proposal interface、构建 exact deployment consequence labels，并开始检验 success/unsafe/handoff factors。现在开始执行 F2，不等待人工确认。
