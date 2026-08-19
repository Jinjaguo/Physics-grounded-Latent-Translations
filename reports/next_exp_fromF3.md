# Next Experiment from EXP_F3：EXP_F4 Deployment-Consistent Consequence Interface

## F3 后主线状态

confirmed-failure benchmark和matched execution成立；F2+F3在共享8个articulated failures上的proposal oracle union为7/8。F3还提供了三个明确、可分别监督的机制：

1. proposal是否到达/保持对象placement；
2. required articulation是否完成；
3. disengage/fallback/handoff是否保留已有进展并允许base policy继续。

F4进入Research North Star的Priority C：定义deployment-consistent consequence labels/interface。F4不训练最终selector，不比较richer-future model，也不回到failure prediction。

## 新科学假设

> 在完全相同的 `confirmed failure → one proposal → real reobservation → standardized disengage/handoff → fresh base-policy continuation` 因果协议下，`placement/progress retention`、`required articulation completion`、`harm/worsening` 与 `handoff compatibility` 会在真实outcomes中独立变化；这些factor能区分final-success标量混合的不同机制，并形成可跨proposal family使用的最小consequence interface。

只有实际label disagreement与机制案例成立，factor才保留。若某factor在新数据中与另一个factor完全等价或没有独立决策作用，就删除或重定义，不能为了“多head”保留。

## 标准化 causal contract

每个 candidate 必须在同一 confirmed-failure snapshot 上独立执行，严格按以下顺序：

```text
restore confirmed failure + clear pending nominal queue
→ execute exactly one frozen recovery proposal
→ capture actual post-proposal observation/state
→ execute standardized open-gripper disengage/settle when proposal did not terminate safely
→ capture actual handoff state
→ call the same fresh base policy from that real state
→ measure exact final task outcome
```

proposal outcome可提前因official success或done终止；不能根据其他candidate的未来结果改变当前candidate。所有task predicates只用于离线label/evaluation，不作为online candidate selector输入。

## 数据与 candidate coverage

- 使用F3的10个task03/task09 failures，并加入F1 audited task05 states，使factor不只服务articulated tasks；
- 新生成至少4个confirmed failures，覆盖placement-only与articulated任务、不同init states和displacement方向；
- proposal set至少包含：full-task feedback、semantic milestone composition、analytic/hybrid closure（仅作为可执行proposal，不用predicate oracle选它）、retrieval或短action correction、structured fallback；
- 对每个state离线执行全部choices以生成counterfactual training/development labels，但保留state-group split信息；
- 报告当前proposal oracle coverage与fallback coverage，继续区分proposal absence和label/predictor问题。

## factor candidates 与精确定义

每条outcome至少保存以下三个时间边界：failure、post-proposal、post-disengage/handoff、post-continuation。候选factor为：

1. `task_progress_retained`：proposal/handoff后已经满足的目标原子是否保持；articulated tasks分别记录`In`和`Close`，非articulated tasks使用对应object-goal atom与mask；
2. `articulation_completed`：仅在任务要求fixture goal时有效；保存mask，不能把不适用任务当负例；
3. `harm_or_worsening`：done-without-success、非有限状态、已满足goal atom回退、物体从目标区域被移出等受支持且可达的实际机制；
4. `handoff_compatible`：从真实post-handoff state调用冻结base policy后，在固定continuation authority内达到official success；
5. `final_success`：作为最终标量baseline/derived target，不冒充factor解释。

如果`handoff_compatible`与post-proposal official success完全重合，F4必须明确证明它没有新增信息并暂时删除，而不是保留一个空head。

## 机制级 representation evaluation

F4无需先追求预测accuracy；首先回答labels是否构成有意义interface：

- 构造factor joint-distribution和pairwise disagreement matrix；
- 列出同一final-success标量但factor vectors不同的实际rollouts；
- 列出相同proposal family在不同states上的factor变化；
- 检查fallback success、harm、handoff是否可独立变化；
- 对每个factor报告task/proposal适用mask、正负样本数和state-group分布；
- 用只看final scalar无法区分、而factor可以区分的具体decision counterexamples量化“为什么factorization不是多个heads”。

这些分析必须从真实intervention traces得到，不能合成标签或用未来oracle信息进入部署决策。

## 支持、否定与下一步分支

支持：至少两个candidate factors在state-grouped数据中有充分正负样本、存在独立变化和明确decision counterexample；handoff label严格对应部署因果链；proposal coverage仍足以支撑后续selector研究。则F5训练matched factorized predictor、scalar verifier和direct selector。

部分否定：articulation factor只在task03/09有意义，则保留mask并将它作为task-conditional factor，不宣称通用；handoff若退化则删除并继续采集需要中途恢复的任务。

否定：大多数outcome只有全0/全1、factor完全共线或proposal coverage因标准化handoff大幅下降。则F5先改变dataset/proposal termination与任务构造，不训练一个必然退化的多head模型。

## F4 完成 artifacts

```text
experiments/EXP_F4/frozen_protocol.json
experiments/EXP_F4/confirmed_failures.jsonl
experiments/EXP_F4/proposals.jsonl
experiments/EXP_F4/rollouts/*.npz
experiments/EXP_F4/outcomes.jsonl
experiments/EXP_F4/factor_schema.json
experiments/EXP_F4/factor_matrix.csv
experiments/EXP_F4/metrics.json
experiments/EXP_F4/run_metadata.json
experiments/EXP_F4/audit.json
reports/EXP_F4_report.md
reports/next_exp_fromF4.md
```

审计必须从step traces重算所有factor、proposal/handoff phase、official success与coverage。F4完成后按真实factor disagreement决定F5模型，而不是预先假定factorization成立。现在立即执行F4。
