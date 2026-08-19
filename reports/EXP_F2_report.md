# EXP_F2 Report：Task-Milestone Feedback Proposal Coverage

> **F5勘误：** legacy snapshot未保存LIBERO placement sampler写入的`model.body_pos/body_quat`。同一F2 failure下matched choices仍共享相同restored variant，真实intervention与相对coverage有效；但fresh env不保证完整等于最初failure-generation world。详见`reports/F1_F4_SNAPSHOT_MODEL_STATE_ERRATUM.md`。

## 实验有效性

```text
new_data_generated: true
new_model_trained: false
new_intervention_executed: true
machine_verifiable_artifacts: experiments/EXP_F2/{frozen_protocol.json,confirmed_failures.jsonl,failures/,rollouts/,outcomes.jsonl,coverage_matrix.csv,metrics.json,run_metadata.json,audit.json}
experiment_id_valid: true
```

F2 新生成 4 个 confirmed-failure checkpoints，并在 4 个 F1 matched checkpoints 与 4 个新 checkpoints 上实际执行 40 条 recovery/fallback controllers、8,688 个控制步。独立审计逐条核对 snapshot、action、phase、cycle、current goal predicates 和 official success，`passed=true`、`discrepancies=[]`。

## 为什么 F2 紧接 F1

F1 在 drawer/task03 与 microwave/task09 上 local/fallback coverage 为 0/4；5-step open-loop proposal 后给 full-task base policy 160 步仍全部失败。竞争解释包括 proposal 太短、handoff budget 不够、full-task prompt 无法恢复 task phase，以及 placement 和 articulated closure 需要不同 controller。F2 没有调 selector 或训练 consequence model，而是引入长时、每 5 步真实重观察的 task-milestone proposal formulation，并设置 equal-total-budget control。

## 实验前冻结的假设与判定

`experiments/EXP_F2/frozen_protocol.json` 在正式 outcome 前声明：

> 对需要把对象放入 articulated receptacle 并关闭机构的 confirmed failures，re-observed task-milestone controllers 应比同预算 full-task replan 增加 oracle coverage。

支持模式是 structured method 在 equal-budget full-task 失败的 state 上成功并增加 union coverage；否定模式是 structured methods 仍为 0，或所有改善都能由同预算 full prompt 复现。

## 数据 provenance 与 split discipline

- matched cohort：F1 的 task03/task09 四个 audited snapshots，只用于同状态新 intervention，没有改写 F1 outcome；
- new cohort：官方 train init index 2、3，各自先施加新的 1 cm initial object offset，再执行 10 wait + 40 个新 base-policy steps，随后由外部 benchmark controller施加并测量 5 cm task-object displacement；
- 新 failure 共 4 个：task03 两个、task09 两个；snapshot restoration 均为 exact-zero test pass；
- 当前全部数据仍是 development，不是 final confirmation；
- retrieval 只读取 Wave-19 train success trajectories；F1/F2 outcome 不进入 retrieval proposal；
- evaluator-only 执行全部 choices，尚无 online selector。

## 新 proposal/control formulation

每个 choice 总预算固定 240 个真实控制步，前 80 标记 proposal phase、后 160 标记 handoff phase；除 retrieval action外，π0.5 每 5 步从实际双相机与 proprioception 重新规划。五个方法为：

1. `extended_full_task_replan`：240 步始终使用原 full-task prompt；这是区分“F1 预算不足”与“task structure 有效”的关键对照。
2. `fixed_semantic_milestones`：前 80 步使用 placement prompt，之后固定使用 closure prompt，不读取 partial goal predicate。
3. `predicate_feedback_milestones`：每个 cycle读取当前时刻的环境 goal predicates；`In=false` 使用 placement prompt，`In=true && Close=false` 使用 closure prompt。这是 simulator/task-predicate proposal upper bound，不是未来 outcome oracle。
4. `adapted_retrieval_feedback`：前 80 步每 5 步重新检索 train-success issue state，并按当前对象与检索对象位置差适配 translation actions；之后使用 predicate feedback handoff。
5. `structured_fallback`：不 reset world；先 open-gripper vertical disengage，再从当前 state 使用 predicate-structured replan。

task03 的 milestones 是 `put bowl in bottom drawer` → `close bottom drawer`；task09 是 `put mug in microwave` → `close microwave`。每个 NPZ 保存 cycle prompt/stage、当前 `In/Close`、action、EEF/object state、proposal/handoff phase 与 success。

## consequence 与 metric

- `In milestone ever`：任一实际 post-action state 满足 BDDL `In`；
- `Close milestone ever`：任一实际 post-action state 满足 BDDL `Close`；
- `final success`：同一步同时满足官方完整 goal；
- method coverage：8 个独立 failure checkpoint 中该方法 final success 的比例；
- oracle local coverage：四个非-fallback choices 的并集中存在成功的 checkpoint 比例；
- structured gain over extended：equal-budget full-task 失败，但 fixed/predicate/retrieval 至少一个成功；
- supported unsafe/invalid：exception、nonfinite 或 done-without-success。

`In/Close` 是当前 simulator state 的可测物理/task predicate，用于机制分解；没有用未执行 proposal 的未来结果。

## 定量结果

| method | success / 8 | coverage |
|---|---:|---:|
| extended full-task replan | 4/8 | 0.500 |
| fixed semantic milestones | **5/8** | **0.625** |
| predicate feedback milestones | 4/8 | 0.500 |
| adapted retrieval feedback | 1/8 | 0.125 |
| structured fallback | **5/8** | **0.625** |

整体 decomposition：

| metric | result |
|---|---:|
| confirmed failures | 8（4 F1 matched + 4 F2 new） |
| intervention outcomes | 40 |
| actual control steps | 8,688 |
| oracle local coverage | 6/8 = 0.750 |
| multiple local solutions | 4/8 = 0.500 |
| structured fallback success | 5/8 = 0.625 |
| structured gain over equal-budget full task | 2/8 |
| outcomes reaching `In` | 32/40 |
| outcomes reaching `Close` | 19/40 |
| supported unsafe/invalid outcomes | 0/40 |

leave-one-method-out marginal union coverage：extended full task 1/8，fixed semantic 1/8，predicate 0，adapted retrieval 0。task03 coverage 为 4/4，task09 为 2/4；task03 fallback 4/4，task09 fallback 1/4。

## checkpoint-level 机制结果

- `matched_task03_instance00/01`：extended、fixed、predicate 都成功；第二个 state 的 adapted retrieval 也成功。F1 的 160-step协议在同 states 上全失败，而 F2 extended 分别用 177/187 步成功，证明 F1 对 drawer 的主要限制之一确实是 continuation budget。
- `new_task03_instance02`：只有 extended local 成功，fixed/predicate/retrieval 都已经达到 `In` 但无法 `Close`；structured fallback成功。task structure 在这个 state 反而伤害 closure timing。
- `new_task03_instance03`：extended/fixed/predicate/fallback成功，retrieval达到 `In` 但未关闭。
- `matched_task09_instance00`：所有方法都没达到 `In`，仍是 placement/proposal coverage failure。
- `matched_task09_instance01`：extended 已达到 `In` 但未关闭；fixed 与 predicate 在 219 步成功，是 structured gain 正例；fallback同样达到 `In` 但未关闭。
- `new_task09_instance02`：extended 达到 `In` 但未关闭；fixed 在223步成功，structured fallback在233步成功；predicate 达到 `In` 仍未关闭。这是第二个 structure-over-budget 正例。
- `new_task09_instance03`：除 extended外，structured/retrieval 都达到 `In`，但全部无法 `Close`；这是清楚的 closure bottleneck。

## 成功案例与失败案例

最强成功案例是两个 microwave state：同样 240-step预算下 full-task prompt已经把 mug 放入 microwave但没关门，而 fixed placement→closure prompt 完成完整任务。这证明 task factor/phase decomposition可以实际改变动作与最终 outcome，不只是延长 horizon。

最重要的负例是 predicate upper bound 没胜过 fixed timing：它仅4/8，且在 `new_task09_instance02` 与 task03 instance02 中都在 `In` 后调用 closure prompt但失败。exact current predicate只解决“何时换 prompt”，不保证 closure proposal本身有接触与控制 coverage。

adapted retrieval 只有1/8；多条轨迹到达 `In` 后卡在 closure。用对象位置差修正 raw action translation没有恢复 fixture handle/contact geometry，否定了该 retrieval adaptation。

## competing explanations 判定

1. **F1 handoff budget不足：支持但不是全部解释。** drawer 的 matched 0/2 在 177–187 步变为 extended 2/2；但 microwave extended仍0/4。
2. **semantic task decomposition增加 coverage：支持。** fixed semantic 在两个 extended-fail microwave states成功，带来 1/8 独立 marginal coverage和2个 structured gain cases。
3. **exact predicate feedback优于 fixed switching：否定。** predicate 4/8，低于 fixed 5/8；current `In` 并不足以生成可靠 closure action。
4. **state-adapted raw retrieval修复 open-loop mismatch：否定。** 1/8，且大量 placement→closure失败。
5. **当前 fallback已可靠：未建立。** task03 4/4但 task09 1/4；fallback仍依赖同一个不可靠 closure controller。

## 与 F1 相比的新科学信息

F1 只能说“当前 5-step proposal set在 articulated tasks上无解”。F2 将原因拆开：

- task03 并非缺少可恢复动作，主要需要更长控制预算；
- task09 的两个 state需要显式 semantic phase decomposition，而另两个仍无完整 coverage；
- 32/40 到达 `In`、只有19/40到达 `Close`，说明 placement和articulation closure是独立、可监督的 consequence mechanisms；
- 当前 predicate feedback不是 final proposal generator；真正缺少的是可执行 closure skill，而不是更准的 `In` classifier；
- fallback不能与 closure proposal共享同一个结构性弱点，否则“选择 fallback”也无效。

这些结果仍属于 proposal substrate，而不是 factorized consequence model 的证据。

## bug、smoke 与无效运行

- `experiments/EXP_F2_smoke/`：π0.5 server 已退出，client在 localhost reconnect循环等待；人工中止前没有生成 rollout，消耗0 EXP。
- `experiments/EXP_F2_smoke2/`：服务恢复后完成3个 checkpoint×5 methods、150步的10-step接口验证，不进入正式 metric。
- 正式 `EXP_F2` 没有 invalid rollout、exception、nonfinite或restore mismatch。

没有因为 smoke、server重启或短预算 gate推进编号。

## 审计、泄漏与不确定性

`experiments/EXP_F2/audit.json` 检查8 snapshots、40 rollouts、8,688 steps，重算 method/union/fallback coverage、milestone counts和structured gains，所有值与 `metrics.json` 一致。

每个 failure 是独立统计单位；5个 methods不是40个独立 failure samples。只有2 tasks×4 states，task heterogeneity很大，因此不做跨机器人/跨任务总体显著性外推。F2的可靠结论限于：在这些 confirmed-failure states上，structured prompts产生两个 matched mechanism wins，并将当前 union coverage提高到6/8。

## 支持与不支持的 claim

支持：长时re-observed proposal显著扩大当前 benchmark headroom；semantic milestone structure在两个 microwave states有同预算因果增益；placement与closure failure可以从真实 predicates分别诊断。

不支持：proposal coverage已经足够广、predicate oracle是好 selector、adapted retrieval有效、fallback已校准、factorized consequences优于scalar、unseen-state selector、selected-only闭环或最终系统完成。

## artifacts 与磁盘

- runner：`scripts/experiments/run_exp_f2_task_milestone_proposals.py`
- audit：`scripts/experiments/audit_exp_f2.py`
- protocol/data：`experiments/EXP_F2/frozen_protocol.json`、`confirmed_failures.jsonl`、`failures/`
- rollout/outcome：`experiments/EXP_F2/rollouts/`、`outcomes.jsonl`、`coverage_matrix.csv`
- aggregate/runtime/audit：`metrics.json`、`run_metadata.json`、`audit.json`

完成后磁盘剩余846 GB；没有删除任何旧 evidence。

## F2 结论

F2 有效提高了 proposal coverage，但 task09 仍只有2/4。主瓶颈已经从笼统的“proposal太短”收缩为**缺少可靠 articulated closure recovery skill**。在解决它之前训练 selector仍会把 proposal absence误归因于 consequence prediction。
