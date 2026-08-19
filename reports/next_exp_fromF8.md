# Next Experiment from EXP_F8：EXP_F9 Factorized Risk Selector on Mixed Recovery/Fallback States

## F8后的主线状态

F8在新6-state cohort上达到proposal-set coverage 6/6，但recovery/semantic都饱和；强制acquisition gate被否定。F6–F8现有canonical full-state数据提供18个unique failure groups和162个真实outcomes，可用于训练/development，但不能再作为untouched confirmation。下一步可以进入consequence selector，同时必须引入真正包含fallback-correct states的未见benchmark。

## 新科学假设

> 一个显式预测`object/phase readiness`、`harm/worsening risk`和`handoff compatibility`并以hard harm veto与fallback abstention组合的factorized interface，能在未见mixed-mechanism failures上避免有害local recovery、选择可续接proposal，并在没有可信local proposal时选择safe fallback；matched scalar final-success predictor无法同样清楚地区分“无proposal”“有害proposal”和“handoff不兼容”。

F9不恢复F5已经被数据否定的coarse articulation head；articulation progress只在object goal满足时作为masked阶段factor。F8 acquisition不进入最终head。

## Training/development data

- 合并F6、F7、F8真实outcomes，按failure ID分组；同一state的不同contract/proposal始终在同一fold。
- 统一decision-time features：双相机视觉block features、robot policy state、task、proposal family/phase描述、first action prefix与budget；禁止post-action observation和outcome。
- 统一labels：proposal object goal/retention、object-masked articulation progress（仅F7/F8可用）、harm/worsening、exact handoff/final result。
- 训练shared-encoder factor model、参数量匹配scalar final predictor和task/proposal prior；用grouped OOF获得calibration与abstention rule。
- factor selector先harm veto，再要求阶段适用factor与handoff下界；若无candidate通过则选fallback。所有rule在新confirmation outcomes前冻结。

## Mixed-mechanism untouched cohort

新生成至少task03/05/09各2个full-state confirmed failures（优先instance12/13），并对每个基础state预先构造两类可复现机制：

1. `recoverable displacement`：与当前failure construction同尺度，预期至少一个local proposal存在；
2. `severe but physically valid displacement/occlusion or object-out-of-local-workspace`：不reset world，预期local proposals不可信，safe disengage+replan是正确决策。

严重机制必须通过真实simulator state construction和robot interventions验证；不能靠事后看oracle outcome挑state。fallback correctness拆成：safe disengagement、无harm、状态可交给replanner，以及短budget task success（若有），而不是强迫后者为1。

## Causal evaluation顺序

1. 冻结model、feature transform、threshold、factor rule与scalar/direct baselines；
2. 从每个新failure只保存decision-time inputs并输出唯一choice/fallback；
3. 只执行factor selector选中的branch，重新观察并记录handoff；
4. 另外执行scalar selector与prior各自的selected-only matched cohort，不能共享outcome；
5. selected execution全部完成后，才从snapshot离线执行all-choice oracle matrix，诊断coverage、regret和正确fallback；
6. 分开报告proposal-set无有效local、valid local存在但selector错选、recovery成功但handoff失败、fallback安全但任务未在短budget完成。

## 必须比较

- factorized risk selector；
- matched scalar final-success selector；
- no-harm-head、no-handoff-head、scalarized-factor ablations；
- task/proposal prior；
- oracle local coverage与oracle fallback-correct label（仅诊断，不能进入decision）。

核心指标是selected-only task success、harmful selection、fallback precision/recall、unsafe false-local rate、coverage-conditioned regret和handoff failure，不只报AUROC。若新cohort仍全local可恢复，F9内继续修failure construction，不能用全成功写selector结论。

## 完成artifacts

```text
experiments/EXP_F9/frozen_protocol.json
experiments/EXP_F9/training_manifest.json
experiments/EXP_F9/feature_schema.json
experiments/EXP_F9/checkpoints/
experiments/EXP_F9/oof_predictions.csv
experiments/EXP_F9/confirmed_failures.jsonl
experiments/EXP_F9/decision_inputs/
experiments/EXP_F9/decision_manifest.jsonl
experiments/EXP_F9/selected_rollouts/
experiments/EXP_F9/selected_outcomes.jsonl
experiments/EXP_F9/oracle_rollouts/
experiments/EXP_F9/oracle_outcomes.jsonl
experiments/EXP_F9/metrics.json
experiments/EXP_F9/run_metadata.json
experiments/EXP_F9/audit.json
reports/EXP_F9_report.md
reports/next_exp_fromF9.md
```

F9只有在mixed cohort真实生成、choices在outcome前冻结、selected-only branches执行并reobserve、post-selection oracle完成且审计通过后才完成。现在立即执行，不等待确认。
