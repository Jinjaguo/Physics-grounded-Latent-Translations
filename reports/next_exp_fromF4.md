# Next Experiment from EXP_F4：EXP_F5 Frozen Factor Predictor and Selected-Only Unseen Evaluation

## F4 后主线状态

benchmark、proposal headroom和deployment label contract已成立。F4的最小learned interface为：

```text
proposal_object_goal
proposal_articulation_goal (task mask)
handoff_compatible (proposal-not-complete mask)
```

retention heads已因全正而删除；harm只有2个正例，保留作evaluation constraint，不训练主head。F5首次进入Priority D：对unseen confirmed failures做predictor/selector evaluation，并执行真正selected-only branch。

## 新科学假设

> 仅使用decision时可得的confirmed-failure observation/state和proposal identity/parameters，shared factorized predictor能在unseen failure groups上估计object、articulation与handoff consequences；由这些factor构造的冻结decision rule，能比matched scalar verifier、success-only/object-only ablations更可靠地把proposal headroom转化为selected-only recovery，尤其避免“关闭空机构”与“object已就位但需要handoff”两类混淆。

F5不假定该hypothesis成立。若matched scalar同样好或更好，必须报告并在F6改变representation/data，而不是调test threshold。

## 输入与禁止信息

decision input只包含：

- confirmed-failure boundary保存的agentview/wrist image；
-当时policy state/proprioception；
- task ID/goal mask；
- proposal family ID、冻结预算与已知proposal参数。

不能输入post-proposal state、已执行action trace、goal predicate outcome、其他candidate结果或未来base-policy success。proposal method ID是部署时已知的candidate定义，不是outcome leakage。

## matched models

在F4的16个failure groups上进行严格state-grouped cross-validation：

1. shared-encoder factorized predictor：object/articulation/handoff三个masked heads；
2. matched-capacity scalar verifier：直接预测exact-contract final success；
3. object/success-only ablation；
4. factorized去掉articulation；
5. factorized去掉handoff；
6. method-prior/task-prior baseline。

图像使用冻结、预声明的低维视觉特征或固定encoder；不能在confirmation outcome上选择特征。模型容量、训练groups、proposal信息和优化预算尽量匹配。threshold/calibration只用F4 grouped out-of-fold predictions决定。

## factorized decision rule

对无articulation任务，proposal-complete probability由object factor决定；articulated任务由object与articulation组合。未完成proposal的final utility再结合handoff factor。具体组合、fallback threshold与tie-break必须在查看F5 outcome前冻结到protocol和decision manifest。

fallback是第六个实际choice，不reset world。如果local最大可信度低于F4-development threshold，选择fallback；否则选一个local proposal。F4没有no-local/fallback-only state，因此F5必须把fallback calibration证据限制在实际出现的confirmation机制，不能制造标签。

## untouched confirmation protocol

- 在模型、feature transform、threshold和decision rule全部冻结后，生成至少6个新confirmed failures：task03/task05/task09各2个未使用init states和新displacement directions；
- 对每个state先保存所有model predictions和唯一selected choice；
- 从未被任何candidate执行过的snapshot执行selected-only branch，保存proposal→reobserve→handoff→continuation完整trace；
- selected-only完成后，才从原snapshot离线执行全部choices，获得oracle coverage、factor labels和selector headroom；
- confirmation outcome不能反向改变模型、threshold或candidate definitions。

## evaluation decomposition

- 每个factor的masked Brier、AUROC/average precision（正负均存在时）、calibration与group bootstrap区间；
- final-success prediction的Brier/log loss；
- oracle coverage、selected-only success、regret与fallback choice；
- error attribution：proposal absence、factor prediction、decision-rule、execution、handoff；
- harmful selection与“close empty fixture”选择次数；
- matched scalar、object-only、no-articulation、no-handoff和prior controls；
- F4 grouped OOF与F5 untouched confirmation分开报告。

样本量小则报告exact counts和state-level cases，不伪造显著性。

## 支持、否定与下一步

支持：factor heads在unseen states有非平凡校准，factorized selector提高selected success或减少可解释harm/closure-object混淆，并在decision manifest证明selected-before-outcome。F6扩展到repeated recovery loop。

否定：factorized不优于scalar、某head不能泛化、selector未利用oracle headroom或fallback误选。F6必须根据error decomposition改变模型族、data construction或decision formulation，不能重复调hidden size/threshold。

## F5 完成 artifacts

```text
experiments/EXP_F5/frozen_protocol.json
experiments/EXP_F5/training_split.json
experiments/EXP_F5/checkpoints/
experiments/EXP_F5/oof_predictions.csv
experiments/EXP_F5/confirmed_failures.jsonl
experiments/EXP_F5/decision_manifest.jsonl
experiments/EXP_F5/selected_rollouts/*.npz
experiments/EXP_F5/oracle_rollouts/*.npz
experiments/EXP_F5/oracle_outcomes.jsonl
experiments/EXP_F5/metrics.json
experiments/EXP_F5/run_metadata.json
experiments/EXP_F5/audit.json
reports/EXP_F5_report.md
reports/next_exp_fromF5.md
```

现在立即执行F5；模型必须先冻结，confirmation data随后生成。
