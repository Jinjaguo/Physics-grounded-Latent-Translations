# Next Experiment from EXP_F12：EXP_F13 Factor-Terminal Recovery Option Graph

## F12后的主线状态

F12完成了two-stage recovery loop，但固定80-step feedback使full factor controller只有3/12 success、1 harm，低于segmented static的4/12、0 harm。三次`observed_regression`中至少一次本应继续并成功，一次prefix后fallback本身harm。删除object/progress虽改变动作，却不改变success。失败原因是factor observation与monolithic policy的任意时间切点不对齐，而不是阈值略偏。

## 新科学假设

> Factorized consequences只有在与可终止、可组合的recovery options对齐时才构成有意义interface。以`object acquired/stable`、`object at goal`、`articulation progress/completion`作为option termination和handoff contracts，比固定时长prefix更能支持`intervention → feedback → next option`；同时，fallback必须根据object-contact factor选择stable-place或disengage，而不能统一release-lift。

这不是failure prediction，也不执行多个candidate outcome后再选。每个branch沿一个option graph实际前进，只在自身达到事件终点或budget/安全条件时重新决策。

## Recovery option graph

为task03/09 articulated tasks构造：

```text
confirmed failure
  -> Acquire/Recover option (terminate: stable acquisition or object_goal)
  -> Place option (terminate: object_goal stable K steps)
  -> Articulate option (goal-conditioned analytic/policy control; terminate: joint progress/completion)
  -> base-policy handoff
```

task05使用Acquire/Recover → Place → base handoff。每个option保存entry/exit factors、执行动作、termination reason和snapshot。只有当前branch的真实exit state进入下一个decision。

Fallback根据exit factors分两类：未接触/未持物时disengage-replan；已持物或object unstable时先执行stable-place/hold option，再replan。必须分别报告fallback decision正确性与fallback execution harm/success。

## 实际ablation

在同一新confirmed-failure cohort实际执行：

1. full factor-terminal option graph；
2. no-object-terminal：不在acquisition/object-goal事件handoff；
3. no-articulation-option：object goal后直接base policy；
4. monolithic recovery（F12 static-style control）；
5. scalar option selector：相同options但用matched scalar决定next option。

full与ablations必须共享stage1 support、总action budget和initial recovery。删除factor要改变明确的option transition，否则不算机制ablation。

## Development与冻结

- 用F7 aligned articulation和F8–F12 traces确定事件定义；按full-state group评估每个option entry factor对exit success/harm的条件关系。
- 不用F13 formal outcomes选择稳定窗口、budget或transition rule。
- F13 formal使用未见official instances；所有stage1 decisions先冻结，每个online transition在next option执行前单独落盘。
- 全部selected option graphs完成后才执行static proposal oracle。

## 必须回答

- event-terminal options是否避免F12的transient-regression false stop；
- object/articulation factor删除分别造成哪些错误handoff或遗漏；
- option graph是否修复“proposal set有成功项但initial ranking错误”；
- factor graph相对matched scalar/monolithic是否改善selected success或harm；
- stable-place fallback是否在持物/接触states上优于统一release-lift。

## Pivot规则

- 若option entry根本达不到，瓶颈是proposal generator coverage，F14引入更丰富goal-conditioned proposals；
- 若entry/exit factors可靠但transition仍错，F14学习option-level dynamics，而不是预测长raw trajectory；
- 若full与factor deletion相同，删除无控制价值factor；
- 若fallback继续造成harm，F14独立构建safe placement/restart controller并把task completion与safety分开。

## 完成artifacts

```text
experiments/EXP_F13/{frozen_protocol.json,option_schema.json,confirmed_failures.jsonl,stage1_decisions.jsonl,option_transitions.jsonl,option_traces/,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
reports/EXP_F13_report.md
reports/next_exp_fromF13.md
```

F13只有在新states上真实执行完整option transitions、五个selected branches、post-selection oracle和零discrepancy审计后才占用EXP_F13。
