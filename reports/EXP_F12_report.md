# EXP_F12 Report：Closed-Loop Factor Feedback Recovery

## 实验有效性

```text
new_confirmed_full_state_failures: 12 (6 moderate + 6 severe)
frozen_stage1_decisions: 12
online_prefix_feedback/stage2_decisions: 60
actual_closed-loop_selected_rollouts: 60
strictly_post-selection_oracle_rollouts: 48
selected/oracle_steps: 11540 / 15202
machine-verifiable_artifacts: experiments/EXP_F12/{frozen_protocol.json,confirmed_failures.jsonl,restore_cross_env.jsonl,decision_inputs/,stage1_decisions.jsonl,stage1_prefix_traces/,stage2_decisions.jsonl,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
audit_passed: true
audit_discrepancies: []
experiment_id_valid: true
```

F12在untouched official instances 25/26上产生12个新的confirmed failures。所有stage1 support和initial proposal choices先冻结。每个local branch只执行自己选择的80-step prefix，立即保存真实prefix trace和re-observation，写stage2 decision后才继续执行；任何branch都没有访问其他candidate outcome。全部60个dynamic selected完成后才跑48条static oracle。

## 研究假设与新control formulation

F11的静态factor/scalar在task09出现proposal ranking miss。F12检验：不预先执行所有candidates，只执行一个selected recovery prefix，读取object acquisition/goal/stability、target/joint progress、regression和official success，是否能更可靠地继续、切换semantic或fallback。

为隔离feedback机制，stage1统一使用F10 frozen geometry gate；本cohort它把6个moderate全部放行、6个severe全部fallback。local state的initial proposal由F11 factor score选择。五个实际branch共享同一state、initial proposal和80-step budget：full feedback、删除object、删除progress、static factor（分段后继续同proposal）以及re-observed scalar ranking。

full rule在observed target regression `< -0.10`或joint regression `< -0.05`时fallback；articulated task在object ready但joint progress `<0.05`时切semantic；非articulated task在target progress `<0.05`时切semantic。所有阈值在F12 outcomes之前冻结。

## Oracle coverage

oracle再次给出6/12 local viable、6/12 no-local；6个moderate都有至少一个成功local，6个severe三local全失败。stage1 geometry gate因此precision/recall均1.0。task03两个severe的三local全部harm而fallback无harm；其余severe local失败但不一定harm。state support与proposal safety仍是不同factor。

## Selected-only结果

| controller | success | harm | stage2 switches | oracle-viable success |
|---|---:|---:|---:|---:|
| full factor feedback | 3/12 | 1 | 3 | 3/6 |
| no-object feedback | 3/12 | 1 | 4 | 3/6 |
| no-progress feedback | 3/12 | 1 | 5 | 3/6 |
| static factor, segmented execution | **4/12** | **0** | 0 | **4/6** |
| re-observed scalar feedback | 3/12 | 0 | 2 | 3/6 |

full feedback没有改善static，反而少1次成功并多1次harm。full/no-object/no-progress的最终success完全相同，因此object/progress factors虽然实际改变控制branch，却没有产生必要性或性能证据。

## 机制案例

- `task03 instance25 moderate`：initial semantic prefix被判为regression，三种factor feedback都fallback并失败；static继续semantic成功，oracle三local全成功。scalar re-observation切到ungated也失败。这是明确的early-feedback control regret：短期几何回退不代表full proposal最终失败。
- `task03 instance26 moderate`：initial recovery最终失败；oracle只有ungated成功。所有feedback rules都没有提出ungated，属于stage2 proposal/ranking miss，而不是proposal coverage failure。
- 两个task05 moderate：full继续initial、no-progress切semantic，scalar有时切recovery；所有branches和三种oracle local都成功。factors改变了动作但没有outcome价值。
- `task09 instance25 moderate`：80-step recovery prefix已产生regression；factor rules检测到后fallback，但fallback自身在该state也harm，最终full/no-object/no-progress均harm。static/scalar继续失败但无harm；full-horizon recovery oracle却成功。这里同时暴露固定prefix的不可逆副作用、replan trajectory shift和fallback controller failure。
- `task09 instance26 moderate`：所有controllers成功，no-object切semantic但没有额外收益。

因此F12没有修复F11的static ranking问题。更关键的是，feedback detection可以是正确的——prefix确实regressed——但如果feedback发生在不可逆动作之后，或者fallback本身不安全，检测正确也不能改善outcome。

## Factor representation的含义

F12提供了比静态head ablation更强的接口测试：删除object或progress会真实改变stage2 action。smoke中no-object把recovery切为semantic；formal中no-object多1次switch、no-progress多2次switch。但这些变化没有改变success，说明当前factor定义在80-step控制粒度下是可操作但无效的。

这否定“只要factor可监督并参与rule，就自然形成有意义interface”。有意义的factor还必须与正确action abstraction和termination condition对齐。瞬时target margin适合描述物理状态，却不适合作为任意时刻终止monolithic policy的充分条件。

## Static对照的边界

F12的`static_factor`是不读取feedback、不切method，但仍在80步边界重新规划后继续同method；它不是F11完全不中断的static rollout。task09 instance25中full-horizon recovery oracle成功，而segmented static失败，证明replan boundary/seed本身会改变trajectory。报告不把这项差异误归因于factor selector。

## Fallback诊断

正式6个severe上的stage1 fallback均无harm。task09 instance25 moderate在prefix regression后进入fallback却产生harm；同state的fallback oracle也harm。fallback安全明显依赖world/object contact state，不能把`fallback_disengage_replan`当通用安全动作。正确的fallback decision和fallback execution consequence必须分别建模。

## 无效run与修正

- `experiments/EXP_F12_smoke/`在任何新state前因错误导入不存在的F2 module退出，消耗0个EXP ID并保留。
- 修正为使用F8已导入的policy client后，`EXP_F12_smoke_v2/`在instance27完成2 failures、10 dynamic selected和8 oracle。它验证no-object feedback确实改变stage2 choice，但三local都成功，没有用于formal rule tuning。
- 初版代码在stage2执行完成后才汇总写decision。formal前已修为prefix结束立即保存re-observation和candidate actions、append stage2 choice，再执行stage2；audit检查每个prefix artifact早于对应selected artifact。

## 审计

`scripts/experiments/audit_exp_f12.py`从F10/F11 checkpoints重算12个stage1 support/factor choices，检查60个prefix artifacts与selected trace的action、target、object、joint数组逐项一致，从logged feedback重算full/no-object/no-progress/static rules，从fresh scalar probabilities重算scalar stage2 choice，核验prefix artifact早于selected artifact、decision/selected/oracle总体时间顺序、48-choice oracle矩阵和所有汇总。结果零discrepancy。

## F12结论

F12首次实现了完整的`selected intervention → feedback → re-observe → second decision → handoff/continuation`，证明system-level闭环接口可运行且可审计。但固定80-step feedback controller被实验否定：它比不切换的segmented static更差，且一次增加harm。object/progress factors能改变动作，却没有改善outcome。

下一步不能把80改成60或100反复搜索。应改变proposal abstraction：用factor本身定义option终止事件——acquire直到稳定获得、place直到object goal、articulate直到joint progress——只在物理子目标边界re-observe，避免在monolithic trajectory中把瞬时回退当最终后果。同时fallback必须根据是否持物/接触选择稳定放置或disengage，而不是统一release-lift。
