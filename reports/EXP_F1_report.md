# EXP_F1 Report：Confirmed-Failure Benchmark and Recovery-Proposal Coverage Reconstruction

> **F5勘误：** legacy snapshot未保存LIBERO placement sampler写入的`model.body_pos/body_quat`。同一F1 failure下matched choices仍共享相同restored variant，真实intervention与相对结果有效；但fresh env不保证完整等于最初failure-generation world。详见`reports/F1_F4_SNAPSHOT_MODEL_STATE_ERRATUM.md`。

## 实验有效性声明

```text
new_data_generated: true
new_model_trained: false
new_intervention_executed: true
machine_verifiable_artifacts: experiments/EXP_F1/{confirmed_failures.jsonl,failures/,restoration.jsonl,rollouts/,outcomes.jsonl,coverage_matrix.csv,metrics.json,audit.json,frozen_protocol.json,run_metadata.json}
experiment_id_valid: true
```

EXP_F1 生成了 6 个此前不存在的 confirmed-failure 执行状态、42 条 evaluator-only recovery/fallback intervention、6,453 个实际控制步，并通过独立 artifact 审计。它不是 interface gate 或旧数据重算。

## 科学问题与实验前假设

研究主线要求先把 impending-failure detection 完全移出问题，并先回答 proposal coverage。F1 的问题是：在一个外部控制器已经明确确认失败之后，现有 native proposal、成功轨迹检索、解析式 task-space proposal、semantic recovery proposal 的并集能覆盖多少状态；如果不能，当前 world state 上的 safe disengagement + fresh replanning 是否能作为有效 fallback？

在查看正式结果前，`experiments/EXP_F1/frozen_protocol.json` 固定了五个竞争解释：

- A：旧系统的主要瓶颈是 proposal coverage；预期 native 和 union coverage 都低。
- B：native proposal 已经有广泛 recovery headroom；预期 native coverage 高，新增 family 边际收益小。
- C：单一 family 不广，但异质 family 互补；预期多个 family 有正 marginal coverage。
- D：部分 confirmed failures 应该 fallback 或在当前协议下无法恢复；预期出现 no-local/fallback 或 neither 状态。
- E：handoff 语义会改变标签；预期 proposal-only 与 exact deployment handoff outcome 不一致。

F1 不训练 predictor 或 selector，因为在 proposal headroom 尚不清楚时优化 selector 会混淆“没有可选解”和“没有选到解”。

## 与历史证据的关系

Master Prompt 对 P1–P35 的引用经用户确认是表述错误；历史报告统一位于 `reports/`。本实验完整阅读并采用了该目录中实际存在的 G1–G40 总结及 G31/G33/G35/G40 的可执行接口证据：

- 使用已验证的 MuJoCo integration state、robosuite controller、observable buffer snapshot/restore 原语；
- 使用官方本地 `pi05_libero`，每次请求只依赖当前双视角图像、proprioception、prompt 和显式 noise；
- 不继承 G 系列的旧 confirmation outcome，不把 G33 的 rollback trial 当作新的 recovery evidence；
- G 系列数据仅作为 retrieval proposal 的 train-only source library 和接口实现依据。

F1 的所有 failure checkpoint、proposal execution、handoff continuation 和最终 outcome 都是新执行。

## 代码状态与运行环境

- Git HEAD：`4a11d2935f5a07c103bbcef952bfee74a75b8c08`
- 工作树在 F1 前已经是 dirty：存在用户的 `snapshot.py`/π0.5 server 修改、历史 G artifacts 和本轮新文件。完整 `git status --short` 被原样保存于 `experiments/EXP_F1/run_metadata.json`，没有把 dirty tree 伪装成 clean commit。
- Simulator：LIBERO-10 / robosuite / MuJoCo，off-screen EGL。
- Base policy：本地官方 `pi05_libero` PyTorch checkpoint；server metadata 明确记录 `wave19_explicit_noise=true`、`wave19_policy_seed=190819`。
- 正式命令完整保存在 `run_metadata.json`，核心参数为 tasks `3 5 9`、每任务 2 个实例、failure decision 8、5 cm failure shift、5-step proposal、160-step handoff、5-step replanning。
- 正式完成时间：2026-08-18T06:09:11-04:00。

## 新 confirmed-failure 数据构造

### 任务机制

F1 使用三个实际可运行、物理机制不同的 LIBERO-10 task：

| task | instruction | 被扰动物体 |
|---:|---|---|
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | `akita_black_bowl_1_main` |
| 5 | pick up the book and place it in the back compartment of the caddy | `black_book_1_main` |
| 9 | put the yellow and white mug in the microwave and close it | `white_yellow_mug_1_main` |

每任务使用两个历史 train init index（0、1），但在 base rollout 前对任务对象施加方向不同的 1 cm 初始位移，因此是全新的执行状态而不是旧 episode replay。因为全部 50 个官方 init 在旧系列中已经以不同方式被消费，这些状态明确标记为 F1 development，绝不冒充 untouched final confirmation。

### confirmed failure 的确切定义

每条新 episode 先执行 10 个 wait steps，再执行 8 次当前观测下的无状态 π0.5 policy issue，每次提交 5 个真实控制步，共 40 个 base-policy steps。随后外部 benchmark controller 对当前任务对象的 free joint 施加 5 cm XY 位移，并在 `sim.forward -> observables refresh` 后、下一次 base-policy request 前立即发出 confirmed-failure event。

这是一个已发生并可直接测量的 controller/benchmark disturbance，不是 failure forecast、early trigger 或 learned detector。事件成立条件为 requested displacement 与 actual body displacement 在 `1e-8 m` 内一致。每条事件保存：

- pre/post object pose 与实际 displacement；
- failure 双相机图像和 policy state；
- 之前真实执行的 base-policy actions、EEF/object trajectory 和 success history；
- 完整 `LiberoSnapshot`；
- failure source、criterion、first confirmed boundary、pending queue semantics。

所有 failure 时官方 success 均为 false，pending action queue 被明确清空。

## checkpoint 与 controller/handoff 语义

snapshot 包含 MuJoCo `mjSTATE_INTEGRATION`、环境 `timestep/cur_time/done`、OSC controller state、robot recent buffers、gripper action state和 observable sampling buffers。π0.5 server 不暴露或保留 hidden state；脚本本地 action queue 在 failure 和每次 handoff 都清空。

每个 failure snapshot 进行了两次 matched restore，再执行同一 `DUMMY_ACTION`。6/6 checkpoint 的结果为：

- integration-state repeat max error：0；
- observation repeat max error：0；
- controller payload byte-identical：6/6；
- restore 后与保存 integration state 的误差：0。

primary deployment handoff 被固定为：

```text
confirmed failure snapshot
-> execute exactly one 5-step local proposal (or fallback disengagement)
-> observe actual post-action simulator state
-> discard all pending actions
-> request fresh pi0.5 chunk from actual images/state and full task instruction
-> execute 5 steps, reobserve, repeat up to 160 handoff steps
-> official LIBERO success / environment termination / budget exhaustion
```

没有使用“proposal 后恢复旧 world state”、stale nominal queue、未来 outcome 或执行所有候选后在线反选。全 proposal branching 只存在于离线 evaluator oracle matrix。

## recovery proposal 与 fallback

每个 failure state 在相同 snapshot 下实际执行 6 个 local proposals 和 1 个 fallback：

1. `native_0`、`native_1`：当前 failure observation + 原任务 prompt 下的两个显式-noise π0.5 chunk；
2. `retrieval_nearest`：仅从 Wave-19 train success episodes 构建的 state-standardized nearest-neighbor 10-step chunk，执行前 5 步；
3. `analytic_open_reapproach`：由当前 EEF 到实际任务对象的向量生成 open-gripper task-space reapproach；
4. `analytic_closed_vertical`：closed-gripper vertical lift；
5. `semantic_recovery`：当前 observation 下以“recover the manipulated object ... then `<task>`”为 recovery prompt 的 π0.5 chunk；
6. `fallback_safe_disengage_replan`：open gripper + vertical disengagement，然后从未重置的当前 world state fresh replan。

local proposal 与 fallback 都使用相同后续 handoff budget 和 method-neutral continuation noise seed。retrieval 的旧数据只决定 proposal bytes；F1 outcome 从不进入 proposal 生成。

## consequence 与 metric 定义

- `proposal_immediate_success`：5-step proposal 内官方 success 是否成立；
- `final_success`：proposal 后按上述 fresh handoff 执行时官方 success 是否成立；
- `safe_under_supported_predicate`：没有 execution exception、所有保存数值 finite、且没有 `done=true` 而 official success=false；
- `oracle local coverage`：一个 failure state 的 6 个 local proposals 中是否至少一个 `final_success=true`；
- `fallback success`：明确 fallback action 后按相同 current-state fresh handoff 是否成功；
- `harmful relative to fallback`：同一 failure 下 fallback 成功而该 local proposal 失败，仅用于 evaluator diagnosis；
- family coverage：该 family 在多少独立 failure checkpoint 上至少有一个成功 proposal。

F1 没有把 contact force 自行阈值化成“unsafe”，因为当前环境没有项目定义的 force safety predicate。报告只使用环境实际支持的 exception/nonfinite/done-without-success 定义，不扩张成通用安全 claim。

## 主要定量结果

### checkpoint-level decomposition

| failure | task mechanism | local successful proposals | successful families | fallback |
|---|---|---:|---|---:|
| task03_instance00 | bowl → drawer + close | 0/6 | none | 0 |
| task03_instance01 | bowl → drawer + close | 0/6 | none | 0 |
| task05_instance00 | book → caddy | 6/6 | native, retrieval, analytic, semantic | 1 |
| task05_instance01 | book → caddy | 4/6 | retrieval, analytic, semantic | 1 |
| task09_instance00 | mug → microwave + close | 0/6 | none | 0 |
| task09_instance01 | mug → microwave + close | 0/6 | none | 0 |

### aggregate

| metric | result |
|---|---:|
| confirmed failures | 6 |
| actual intervention outcomes | 42 |
| local interventions | 36 |
| executed simulator control steps | 6,453 |
| oracle local coverage | 2/6 = 0.3333 |
| multiple-local-recovery fraction | 2/6 = 0.3333 |
| no-local but fallback succeeds | 0/6 |
| neither current local nor fallback succeeds | 4/6 = 0.6667 |
| fallback success | 2/6 = 0.3333 |
| native family coverage | 1/6 = 0.1667 |
| retrieval family coverage | 2/6 = 0.3333 |
| analytic family coverage | 2/6 = 0.3333 |
| semantic family coverage | 2/6 = 0.3333 |
| each family's leave-family-out marginal coverage | 0 |
| proposal-immediate successes | 0/42 |
| deployment handoff successes | 12/42 |
| binary labels changed by handoff | 12/42 |
| harmful relative to successful fallback | 2 local outcomes |
| supported unsafe/invalid outcomes | 0/42 |

以 checkpoint 为独立单位、task 为 cluster，样本只有 3 个 task × 2 states，不支持宽泛总体显著性结论。最重要的不确定性不是 binomial standard error，而是完全的 task heterogeneity：book 是 2/2，drawer 和 microwave 都是 0/2。因此 F1 只支持“当前三机制开发 benchmark 上 coverage 强烈依赖任务结构”，不外推到全部 LIBERO 或真实机器人。

## 成功与失败案例

### 成功案例：task05_instance00

六个 local proposals 和 fallback 全部最终成功，但没有任何 proposal 在 5 步内完成任务；101–124 个真实 handoff steps 后才达到官方 success。这里 proposal identity 几乎不重要，说明 book failure 处于广泛可继续区域。把 5-step proposal outcome 当标签会把 7 条成功链全部错标为失败。

### 可区分案例：task05_instance01

两个 native proposals 都在 160-step handoff 内失败；retrieval、两个 analytic 和 semantic proposals 分别在 119、130、141、126 handoff steps 后成功，fallback 也在 136 步成功。这证明“当前 observation 继续采两次 native chunk”不等同于拥有 proposal headroom；异质 proposal 至少在一个状态改变了可达 outcome。另一方面，四个非-native family 相互冗余，所以还不能声称某个 family 有独立边际 coverage。

### 结构性失败：task03 与 task09

4 个 failure state 的 28 条选择全部执行完整 proposal + 160-step handoff，数值有限且未触发 invalid predicate，但没有 official success。它们共同需要“把对象放入 articulated receptacle 并关闭机构”，而当前 5-step局部 action 后仅给 full-task base policy 继续，无法建立任何 oracle headroom。现有 evidence 不能区分：

- proposal 没有进入正确的 subtask/milestone basin；
- 160-step handoff 对多阶段 articulated task 不足；
- π0.5 full-task prompt 在扰动状态下无法重新建立阶段；
- 5 cm object shift 造成当前 proposal families 没覆盖的接触/可达性问题。

因此不能把 4 个状态解释成“物理上不可恢复”，只能判定“当前 proposal + fallback formulation 无解”。

## hypothesis 判定

- A（proposal coverage 是主要瓶颈）：**支持**。union 只有 2/6，native 只有 1/6；selector 尚无足够 headroom。
- B（native 已广泛覆盖）：**否定**。native family 在 5/6 failure 上没有成功 proposal。
- C（异质 family 广泛互补）：**弱支持但未建立**。task05_instance01 中非-native proposal 修复了 native failure，但 retrieval/analytic/semantic 彼此冗余，所有 leave-family-out marginal coverage 都为 0。
- D（需要 fallback/不可局部恢复状态存在）：**部分支持**。4/6 对当前 local 与当前 fallback 都无解；但 F1 没产生“local 无解而 fallback 成功”的状态，当前 fallback 自身还不够强，不能据此建立 fallback calibration。
- E（handoff 语义改变标签）：**强支持**。12/42 label 从 proposal-only false 变成 exact deployment handoff success；全部成功都依赖 continuation。

## bug、无效 run 与对结论的影响

1. `experiments/EXP_F1_smoke/`：沙箱禁止 localhost websocket，未产生 rollout；保留为 environment gate，消耗 0 EXP。
2. `experiments/EXP_F1_smoke2/`：单 task、短 10-step handoff 的接口 smoke，生成 7 条链但不进入正式 metric。
3. `experiments/EXP_F1_invalid_tolerance1/`：正式第一次尝试完成 3 个 failures/21 interventions 后，位移验证把 MuJoCo `sim.forward()` 的 `1.64e-9 m` z 数值变化与真实位移错误混同，因 `1e-10 m` 全轴容差停止。该不完整目录原样归档；把容差改为仍可检测真实错位的 `1e-8 m` 后，从头以相同 protocol 重跑。没有从无效 run 选择 proposal、task、seed 或 metric。

这些 gate/bug 没有占用新 EXP ID，也不进入正式 `experiments/EXP_F1/metrics.json`。

## 泄漏与因果审计

- failure 在 recovery method 启动前由外部 benchmark controller 明确给出；没有 early-failure predictor。
- evaluator 执行全部 proposals 只为 offline oracle coverage；没有 deployed selector。
- 每个 proposal 在执行前只含当前 observation/state、task instruction、train retrieval library 或解析式 current-state features。
- 后续真实 outcome、future observations、其他 proposal outcome 不进入 proposal bytes。
- 同一 failure 的 7 个 outcomes 从相同 snapshot 开始；42 条 branch 不作为 42 个独立 failure samples。
- continuation 使用相同 seed schedule但实际 post-proposal observation，不 teacher-force future state。
- 当前 cohort 是 development，不是 final confirmation；旧 Wave-19 train retrieval data 与新 F1 outcome 的角色明确分离。

独立 audit `experiments/EXP_F1/audit.json` 重算全部主要 metric，核对 42 个 rollout 的 action/phase/success trace、6 个 failure displacement、6 个 snapshot 和 6 个 restoration rows，`discrepancies=[]`、`passed=true`。

## 与上一实验系列相比的新信息

G40 证明过 task05 上 rollback/retry 的强 closed loop，但无法说明真实 proposal set 在跨机制 confirmed failures 上是否有解。F1 首次把 trigger 问题完全移除，并对三个机制建立相同 failure → proposal → post-state → fresh handoff causal label。新信息是：

1. task05 的强结果不能外推到 articulated drawer/microwave；
2. current proposal union 的主要限制是 coverage，而不是 selector；
3. short proposal 本身不是有效 consequence label，必须包含真实 handoff；
4. fallback 不能只是“安全退开再给 full prompt”，它在两个 articulated tasks 上同样失败；
5. proposal generator、handoff 和 fallback 现在能从 artifacts 中分别诊断。

## 当前可支持与不可支持的 claim

可支持：F1 benchmark 能复现 confirmed-failure checkpoints；多 family proposal 与 fallback 能从完全相同状态真实执行；deployment-consistent handoff 明显改变 recovery label；当前 proposal coverage 在所测 task 之间高度不均，尚不足以训练主 selector。

不可支持：factorized consequences 的价值、unseen-failure selector generalization、fallback calibration、broad local recovery、selected-only deployment、multi-cycle recovery、真实机器人安全、或 richer-future `Only` claim。F1 没有训练 consequence model，也不应提前声称这些结论。

## artifact 索引与磁盘

- 实现：`scripts/experiments/run_exp_f1_confirmed_failure_coverage.py`
- 独立审计：`scripts/experiments/audit_exp_f1.py`
- frozen design：`experiments/EXP_F1/frozen_protocol.json`
- failure manifest / snapshots：`experiments/EXP_F1/confirmed_failures.jsonl`、`experiments/EXP_F1/failures/`
- restore evidence：`experiments/EXP_F1/restoration.jsonl`
- intervention traces：`experiments/EXP_F1/rollouts/`
- outcome matrix：`experiments/EXP_F1/outcomes.jsonl`、`experiments/EXP_F1/coverage_matrix.csv`
- aggregates / environment：`experiments/EXP_F1/metrics.json`、`experiments/EXP_F1/run_metadata.json`
- audit：`experiments/EXP_F1/audit.json`

完成 F1 后单次磁盘检查显示剩余 846 GB，高于约 200 GB 下限；没有删除任何旧 artifact。

## F1 结论

F1 有效完成了 Research North Star 的第一个 benchmark 原语，但没有解决 proposal coverage。当前最重要的事实是：**对于 exact confirmed-failure states，book task 有充足且冗余的 local/fallback headroom，drawer 与 microwave task 则完全没有；因此下一步必须改变 proposal generator/control formulation，而不是优化 selector 或开始 factorized-vs-scalar 比较。**
