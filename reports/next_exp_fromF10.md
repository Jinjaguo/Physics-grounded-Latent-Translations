# Next Experiment from EXP_F10：EXP_F11 Outcome-Supervised Proposal-Set Viability and Structured Ranking

## F10后的主线状态

F10的deployable visual gate在12个新states上把无support scalar的harm从2降到0、unsafe false-local从7降到1，但产生1个真实false-fallback。更重要的是，task09 instance19在几何上supported却没有任何成功local proposal，说明“EEF-object够近”不是我们真正需要预测的量。factorized + visual成功3/12，matched scalar + visual成功4/12，现有factor ranking仍未建立优越性。

## 新科学假设

> confirmed-failure后的fallback问题应表示为“冻结proposal generator在当前state上是否至少覆盖一个有效local intervention”，而不是距离阈值。以真实oracle interventions监督的proposal-set viability是独立、可诊断的consequence factor；将其与proposal-specific object/stability/direct/handoff factors组合，并用保留空间结构的视觉encoder学习，能同时减少几何proxy造成的false-support和视觉grid造成的false-fallback。若同一encoder、数据和support gate下的structured factor selector仍不优于matched scalar，就应收缩factorization claim，而不是继续添加heads。

这仍以confirmed failure为输入，不加入failure prediction、nominal probe或early trigger。

## Development data与labels

- 汇总并审计F6–F10所有unique full-state failures；同snapshot/同failure的selected与oracle重复只保留一份method outcome，split按official instance/full-state group。
- state-level `proposal_set_viable = any(local proposal final_success)`，只由真实local interventions定义。F10 task09 instance19因此是负例，task05 instance18是正例，不再由距离proxy错误标注。
- proposal-level独立labels继续保留明确语义：object acquired/retained、no harm/no regression、direct task completion、masked handoff compatibility；joint/articulation progress只在适用task上监督。
- 距离只作为development diagnostic/ablation，不进入deployable input，也不作为ground-truth viability。

## 新representation与matched models

采用双相机保留二维feature map的预训练visual encoder（优先使用环境可复现的官方torchvision ResNet18 weights；若本地无权重则从官方source获取并记录），拼接robot state、task和proposal action prefix。冻结或小规模fine-tune策略只由grouped development OOF选择，不用F11 confirmation cohort。

训练三类matched模型：

1. state-level visual proposal-set viability gate；
2. proposal-level structured factors，并用显式task semantics组合选择；
3. 相同encoder、参数规模和输入的direct scalar final-success baseline。

必须报告grouped OOF calibration、leave-experiment/leave-instance generalization，以及每个factor的AUROC/Brier/label prevalence。若某个factor仍低于随机或不改变任何选择，删除或重新定义，不能保留装饰性head。

## 正式selected-only protocol

使用未参与F6–F10、smoke或model selection的official instances 20/21，仍构造task03/05/09各两个moderate及paired 30 cm severe，共12 states。所有visual embeddings、viability probabilities、factor/scalar predictions和choices先冻结，再实际执行：

1. structured factor + outcome-supervised visual viability；
2. matched scalar + 同一visual viability；
3. viability-only + development proposal prior；
4. scalar without viability；
5. F10 geometry-trained visual gate + scalar（frozen ablation）。

全部60个selected完成后才执行48-choice oracle。oracle只诊断proposal coverage、selector regret和fallback correctness，不进入正式decision。

## 必须区分的失败

- `proposal_set_viable=false`且所有local失败：proposal coverage failure，正确动作为fallback；
- viability false-fallback但oracle有成功local：state-factor/generalization failure；
- viability正确且oracle有成功local、selected失败：proposal ranking failure；
- selected recovery完成但base continuation失败：handoff failure；
- fallback无harm但任务未完成：安全退出成功、task recovery未完成，分别报告。

## 核心判据与pivot规则

- outcome-supervised viability必须相对F10 geometry visual gate降低oracle-grounded false-support或false-fallback，且不能用formal outcomes调阈值；
- structured与scalar必须共享visual encoder、viability gate和proposal set，才允许归因representation差异；
- 若structured仍不优于scalar，F12不再重复hidden size/threshold搜索，而转向显式proposal generator diversity或把已证实有用的support/handoff接口保留、删除无价值factor；
- 若viability主要受proposal set变化影响，则F12把proposal generator identity/budget作为condition输入并收集跨generator coverage data。

## 完成artifacts

```text
experiments/EXP_F11/{frozen_protocol.json,training_manifest.json,feature_schema.json,checkpoints/,oof_predictions.csv,confirmed_failures.jsonl,restore_cross_env.jsonl,decision_inputs/,decision_manifest.jsonl,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
reports/EXP_F11_report.md
reports/next_exp_fromF11.md
```

F11只有在新的confirmed-failure states、真实selected-only interventions、post-selection oracle和零discrepancy审计全部完成后才占用EXP_F11。
