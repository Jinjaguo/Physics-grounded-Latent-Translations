# Next Experiment from EXP_F5：EXP_F6 Full-State Short-Horizon Discriminative Benchmark

## F5 后主线状态

F5完成selected-only单决策链，但6个confirmation states上所有selector都6/6，representation的selection价值不可辨。legacy benchmark缺少model body placement；200-step proposal +25-step disengage +160-step continuation也过于宽松，OOF threshold退化为0。object/handoff heads泛化，而articulation AUROC仅0.471；proposal ID不足以描述真实closure action/contact content。

## 新科学假设

> 在完整保存model+integration+controller state的canonical failures上，较短的`80-step proposal → reobserve/disengage → 80-step continuation`会产生可重复candidate disagreement与no-local/fallback cases；显式保存decision-time已知的proposal first action chunk与phase参数，能把“proposal做什么”与state consequence对齐，为后续articulation/handoff predictor提供比method ID更合理的输入。

F6重建benchmark和labels，不沿用F5模型做confirmation claim，也不调F5 threshold。

## Canonical cohort

- 复用F5的6个full-state snapshots；
- 新生成task03/task05/task09各2个，共6个新full-state failures（init 8/9或下一可用indices）；
- 每个snapshot必须保存`model.body_pos/body_quat`、integration、controller、observable；
- 每个state在不同seed fresh env中restore，integration、model arrays、policy state、goal predicates与模型使用的视觉features必须一致；
- legacy F1–F4 snapshots不进入F6 canonical cohort。

## Short-horizon contract

每choice最多80-step proposal；未成功则统一20-step open settle +5-step lift；随后同一base policy最多80步。choices为：full-task；fixed 40-step placement→40-step closure/object；explicit recovery；40-step retrieval→40-step recovery；closure-specific prompt（articulated）/task05 recovery skill；physical fallback disengage→replan。

每个proposal执行前保存decision-time已知first 10-action chunk或analytic command、prompt/phase schedule、budget与family mask。不能保存后续feedback actions作为decision input。

## F6需要回答

- proposal-direct、handoff-rescue、local oracle与fallback coverage；
- 每state成功method数量和真正可区分selector的states；
- object/articulation/handoff joint labels是否仍独立；
- no-local states是否出现，fallback是否安全/成功；
- proposal first-chunk content与articulation outcome是否比method/task prior更有分组关联；
- close-empty-fixture、object-only、handoff-rescue与regression案例。

若12/12仍几乎所有local methods成功，F7改变failure mechanisms而非再缩一点步数；若coverage太低，F7改proposal generator而不训练selector。

## 完成artifacts

```text
experiments/EXP_F6/frozen_protocol.json
experiments/EXP_F6/confirmed_failures.jsonl
experiments/EXP_F6/restore_cross_env.jsonl
experiments/EXP_F6/proposals.jsonl
experiments/EXP_F6/proposal_inputs/*.npz
experiments/EXP_F6/rollouts/*.npz
experiments/EXP_F6/outcomes.jsonl
experiments/EXP_F6/factor_matrix.csv
experiments/EXP_F6/metrics.json
experiments/EXP_F6/run_metadata.json
experiments/EXP_F6/audit.json
reports/EXP_F6_report.md
reports/next_exp_fromF6.md
```

F6只有在新full-state data、actual short interventions、cross-env restoration与disagreement metrics都落盘后才完成。现在立即执行。
