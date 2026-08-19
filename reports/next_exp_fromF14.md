# EXP_F15：Failure-mechanism recovery proposals with real proposal regeneration

## 主线复盘与瓶颈分类

F14 后重新对照 Research North Star 与 F1–F14 结论：

- confirmed-failure benchmark：已基本建立并可 reproducibly restore；
- deployment-consistent consequence semantics：已有分阶段、selected-only、handoff/replan-compatible artifacts；
- selector/generalization：F14 把 6/6 available local headroom 转成 success，本 cohort 不是主要瓶颈；
- fallback execution：F14 找到 bounded safe-exit baselines，但 fallback 没有 task success，factor conditioning无增益；
- handoff/continuation：个别 false-fallback oracle 会在 continuation 产生 harm，继续独立记录；
- **proposal coverage：当前首要瓶颈，6/12 states 的 3 个现有 local proposals 全部失败；**
- **closed-loop state transition：系统还没有真正允许任意多次 `execute one → reobserve → regenerate proposals`。**

因此 F15 的机制问题是：针对远离原 manipulation region、现有 place/recovery/full-task proposals 都无法覆盖的 confirmed failures，显式的 retrieval/staging recovery proposals 能否提高 oracle coverage，并能否在执行一个 proposal、观察真实 state 后重新生成下一轮 proposals，从而把新增 headroom 转成 selected recovery。

## 不再重复的已否定方向

F15 不会：

- 优化 factor-terminal option graph、option 顺序、stable window 或 60/80/100-step boundary；
- 调 factor/scalar threshold、hidden size、epochs、classifier family 或 score combination；
- 试图证明 factorized predictor 全面优于 scalar；
- 新增 fallback factors 或继续寻找 factor-conditioned fallback 优势；
- 将 severe、moderate、EEF-object distance 或 visual probability 当作 recoverability truth；
- 重新引入 failure prediction、nominal probe 或 impending-failure detection；
- 开始 richer-future/world-model baseline。

## 新 proposal families

在现有 `ungated_place`、`recovery_prompt`、`semantic_full_task` 之外，增加机制上不同而非参数变化的 executable proposals：

1. `retrieve_to_workspace_milestone`：让冻结 VLA 找回当前可见的 task object，并把它带回稳定、可继续操作的 workspace/staging region；不要求单 chunk 完成最终 task。
2. `reacquire_object_milestone`：只要求重新获得并稳定持有 displaced object，执行后必须 reobserve；它是 proposal，不重新作为强制 factor gate。
3. `retrieve_then_goal_conditioned_recovery`：显式 recovery subgoal 后从真实 resulting state 重新调用 goal-conditioned recovery，而不是在 outcome 前预写 option graph。
4. 保留现有 monolithic recovery 与 semantic proposal作为 matched baselines。
5. 无可信 local proposal时使用 F14 的 simple fixed safe-exit baseline；safe stop 仍不算 task success。

proposal prompt/skill 必须只用 task goal、当前部署观测和允许的 robot state，不能使用 simulator object pose。若 VLA prompt 本身无法产生空间检索行为，可使用官方/已有实现中的 image-conditioned milestone proposal机制；任何外部机制必须记录 primary source、借用内容和本项目 claim 差异。

## Development 与 formal split

- development：只在已消费的 F13/F14 oracle-no-local states 上执行新 proposal families，确定它们是否确实改变可执行行为，并冻结 task-specific prompt/skill semantics。这里不得调 F15 formal outcomes。
- smoke：使用新 instance37，验证 multi-cycle snapshot、proposal regeneration 与 audit。
- formal：使用未见 official instances35/36，重新生成 12 个 confirmed-failure states。
- 所有 formal deployment decisions 必须在对应 cycle 的 outcome 前保存；cycle 1 可以使用 cycle 0 的真实 post-action observation，但不能访问未执行 proposals 的 outcomes。

## Selected-only controller

最多允许 3 个 recovery cycles，但不是固定三阶段 graph：

1. 从 confirmed-failure state 生成当前 proposal set；
2. 用 formal cohort 前冻结的 simple proposal-family rule/compact consequence interface选择恰好一个 proposal或 safe exit；
3. 执行唯一 choice；
4. 保存真实 resulting snapshot 和可部署 observation；
5. 若 task success，terminal；若当前 state 已恢复到 nominal/recovery-supported region，重新生成 proposal并选择 monolithic/nominal continuation；若仍有可行 retrieval proposal，进入下一 cycle；若无可信 proposal，safe exit；
6. 每个 cycle 的 decision 必须在该 cycle outcome 之前写盘。

F15 的重点是 proposal mechanism 与 real proposal regeneration，不以复杂 learned selector 为贡献。若 development 显示一个 fixed retrieval family 已足够，优先使用它；compact factors只保留已有独立语义的 `worsening/safety` 与 exact handoff/continuation compatibility，不增加无价值 heads。

## Matched baselines

- one-shot existing monolithic recovery；
- one-shot semantic full task；
- one-shot best new retrieval proposal；
- repeated selected proposal regeneration（最多3 cycles）；
- simple fallback/replan。

所有 baselines 从同一 confirmed snapshot 开始，使用匹配总 action/continuation budget。重复 controller 不能通过更大无限预算获益；报告每 cycle 与总 step 数。

## Oracle 与 required decomposition

selected-only 全部完成后，才从每个 initial formal state 执行所有 old/new proposal families，计算：

- old proposal oracle coverage；
- expanded proposal oracle coverage；
- coverage gain 来自哪个新 family；
- local opportunity 中 selected success；
- no-proposal count；
- selector/route miss；
- proposal execution/control failure；
- cycle 0/1/2 state transition；
- handoff/continuation failure；
- harmful proposal 与 safe-exit rate；
- final task success，safe stop 单独报告。

若 expanded oracle coverage 没提高，F15 否定当前 semantic milestone generator，F16 必须换成机制上不同的 analytic/learned/retrieval controller，而不是调 prompt wording。若 coverage 提高但 selected loop未转化，则 F16 聚焦 unseen selector/generalization。若 selected recovery已转化但多轮 continuation失败，则 F16 聚焦 handoff/state transition。

## 预期 artifacts

- `experiments/EXP_F15/frozen_protocol.json`
- `experiments/EXP_F15/proposal_schema.json`
- `experiments/EXP_F15/development_proposal_outcomes.jsonl`
- `experiments/EXP_F15/confirmed_failures.jsonl`
- `experiments/EXP_F15/cycle_decisions.jsonl`
- `experiments/EXP_F15/cycle_feedback.jsonl`
- `experiments/EXP_F15/selected_outcomes.jsonl`
- `experiments/EXP_F15/oracle_outcomes.jsonl`
- `experiments/EXP_F15/metrics.json`
- `experiments/EXP_F15/audit.json`
- `reports/EXP_F15_report.md`
- `reports/next_exp_fromF15.md`

F15 只有在新 proposal families 产生真实 interventions、post-action observations、coverage comparison 和 selected multi-cycle outcomes后才完成。任何 server、snapshot、prompt interface、logger 或 controller 修复都留在 F15 内，消耗 0 新 EXP ID。
