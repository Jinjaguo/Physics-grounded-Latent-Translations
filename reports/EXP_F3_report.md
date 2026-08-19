# EXP_F3 Report：Articulation-Closure Recovery Skills

> **F5勘误：** F3 legacy source snapshots未保存LIBERO placement sampler写入的`model.body_pos/body_quat`。同一closure checkpoint的六方法matched对照仍共享相同restored variant，机制比较有效；但failure fresh env不保证完整等于最初generation world。详见`reports/F1_F4_SNAPSHOT_MODEL_STATE_ERRATUM.md`。

## 实验有效性

```text
new_data_generated: true
new_model_trained: false
new_intervention_executed: true
new_mechanism_ablation_completed: true
machine_verifiable_artifacts: experiments/EXP_F3/{frozen_protocol.json,confirmed_failures.jsonl,boundary_attempts.jsonl,closure_checkpoints.jsonl,checkpoints/,rollouts/,outcomes.jsonl,coverage_matrix.csv,metrics.json,run_metadata.json,audit.json}
experiment_id_valid: true
```

F3 新生成 2 个 task09 confirmed failures，并在 8 个精确的 `In=true, Close=false` snapshot 上执行 48 条 closure/fallback interventions、7,883 个控制步。另有 2/10 failure attempts 未达到 closure boundary，原样保留在总分母。独立审计恢复了全部 8 个 closure snapshots，核对 10 条 boundary traces、48 条逐步 action/goal/joint/gripper traces，重算全部指标，结果为 `passed=true`、`discrepancies=[]`。

## 为什么 F3 紧接 F2

F2 在 task03/task09 上把 oracle local coverage 提高到 6/8，但 40 条 outcome 中有 32 条到达 `In`、只有 19 条到达 `Close`。这说明对象 placement 与 articulated fixture closure 是不同的执行瓶颈。F3 没有训练 selector，也没有回到 failure prediction；它检验的是：在对象已经放入 drawer/microwave、机构尚未关闭的真实状态上，显式 closure proposal 是否增加 proposal headroom，以及关门动作是否会破坏已经完成的 placement。

## 冻结假设与判定

正式 corrected protocol 声明：

> `In=true, Close=false` 是独立 recovery mode；显式 articulation-aligned closure skill 或 analytic-to-policy hybrid 应在保持 placement 的同时增加 recovery coverage，并应优于共享相同 approach、只反转 push 方向的 mechanism control。

支持要求不仅是 joint 发生变化，还要求：aligned/hybrid 在旧 proposal 未覆盖的 state 上产生 `In && Close`；wrong-direction 不产生同等效果；新 task09 states 上可复现。若只关门却破坏 `In`，则只支持 closure progress factor，不支持完整 recovery。

## 数据、confirmed-failure 与 closure boundary

- F2 matched cohort：F2 的 8 个 audited task03/task09 failures；
- F3 new cohort：task09 官方 init index 4、5，各自加入新的 1 cm initial offset，执行新的 base-policy trajectory，再由外部 benchmark controller 施加并测量 5 cm mug displacement；
- 所有 10 个 confirmed-failure snapshots 都保存完整 MuJoCo integration state、controller、observable 和 fresh-replanning semantics；
- 2 个 F3 新 snapshot 的重复 restoration integration/observation error 均为 0，controller bytes 相等；
- 从每个 failure 重新执行固定 placement→closure controller，只在第一个实际 post-action `In=true, Close=false` 状态保存 closure snapshot；未到达者仍留在 failure denominator；
- 10 个 attempts 中 8 个达到 closure boundary，boundary steps 合计 1,307；`matched_task09_instance00` 与新 `instance04` 未达到 placement milestone。

当前全部数据是 development evidence，不是 final untouched confirmation cohort。环境 goal predicate只用于 evaluator、boundary capture和机制标签，不进入 deployed proposal selection。

## 六种实际执行的 closure choices

每个 closure choice 具有相同 200-step authority，从完全相同的 snapshot 恢复；成功可提前终止。每 5 步策略方法从真实双相机和 proprioception 重新规划。

1. `close_prompt_feedback`：持续用任务特定 closure prompt；
2. `full_task_feedback`：持续用原任务 prompt；
3. `analytic_handle_feedback`：先张爪释放并稳定物体，再执行 lift→collision-clear transit→handle precontact→沿真实 joint direction push；
4. `wrong_direction_handle_control`：与 analytic aligned 使用逐步完全相同的 release/approach actions，只在 contact 后反转 push direction；
5. `analytic_then_close_prompt`：100-step analytic prefix 后由 close prompt feedback 接管；
6. `fallback_disengage_close_replan`：不 reset world，先张爪 disengage，再从当前实际状态调用 closure-aware replanning。

analytic interface从运行时 MuJoCo model读取 `white_cabinet_1_bottom_level`/`microwave_1_microjoint`、真实 axis 和 handle geom；没有猜测 geom/joint ID，也没有直接写 fixture qpos。每个 analytic proposal 都通过机器人 action interface实际施力。

## consequence decomposition

F3 对每个 intervention 分开记录：

```text
placement_retained = final In
closure_achieved = final Close
fixture_joint_progress = final joint qpos - initial joint qpos
final_success = official full BDDL goal
safe_under_supported_predicate = no exception/nonfinite/done-without-success
```

这使“关节方向有效但行程不足”“成功关门但丢失 placement”“proposal根本没有到 placement boundary”不再被混进同一个 success rate。

## 定量结果

### 全部 failure decomposition

| metric | result |
|---|---:|
| confirmed-failure attempts | 10（8 F2 matched + 2 F3 new） |
| exact closure opportunities | 8/10 = 0.800 |
| closure intervention outcomes | 48 |
| closure control steps | 7,883 |
| oracle local closure coverage | 6/8 = 0.750 |
| end-to-end local coverage / all attempts | 6/10 = 0.600 |
| fallback success / closure opportunities | 5/8 = 0.625 |
| placement-loss outcomes | 1/48 |
| Close=true but In=false | 1/48 |
| supported exception/nonfinite/done failures | 0/48 |

### method coverage

| method | success / 8 | coverage |
|---|---:|---:|
| close prompt feedback | **6/8** | **0.750** |
| analytic→close-prompt hybrid | **6/8** | **0.750** |
| full-task feedback | 5/8 | 0.625 |
| structured fallback | 5/8 | 0.625 |
| analytic handle feedback | 2/8 | 0.250 |
| wrong-direction control | 0/8 | 0.000 |

task03 的 local closure coverage 为 2/4；task09 为 4/4。close prompt在 task03 为2/4、task09为4/4；analytic-only在 task03为0/4、task09为2/4。

### articulation mechanism control

独立审计证明，每个 checkpoint 上 aligned 与 wrong-direction 的 pre-push action prefix逐元素完全相等，两组都真实进入 push stage。结果为：

| task | aligned mean joint progress | wrong-direction mean joint progress | aligned official success | wrong success |
|---|---:|---:|---:|---:|
| drawer/task03 | +0.058 | -0.009 | 0/4 | 0/4 |
| microwave/task09 | +1.690 | 0.000 | 2/4 | 0/4 |

aligned相对wrong为2 wins、0 losses、6 task-success ties，并且 aligned/wrong 16 条 rollout 都保持 placement。由此可以严谨支持 joint/handle alignment改变了真实物理 consequence；但 drawer analytic 行程仍不足，不能声称 analytic controller 普遍解决 closure。

## checkpoint-level 成功与失败

- `new_task09_instance03` 是最关键的 proposal-headroom正例：F2 四个 local methods全失败；F3 的 close prompt、full task、analytic-only和hybrid都成功，wrong-direction失败。它为 F2 shared cohort增加一个独立覆盖 state。
- `new_task09_instance02` 上 aligned analytic与三个policy/hybrid方法都成功，wrong-direction失败，验证机制效果不只来自一个实例。
- `matched_task09_instance01` 与 F3 new `instance05` 上 analytic joint progress很大但略未越过官方close阈值；close prompt/full/hybrid成功。这表明“正确关节方向”必要但不充分，policy接管可补足contact/行程。
- `matched_task03_instance01` 与 `new_task03_instance03` 被 close prompt/hybrid恢复；后者 full-task失败，说明closure-specific prompt比笼统任务replan更可靠。
- `matched_task03_instance00` 与 `new_task03_instance02` 已经 placement，却没有任何 local closure或fallback成功，是 proposal-present-but-insufficient 的明确负例。
- `matched_task09_instance00` 与 F3 new `instance04` 连 closure boundary 都未到达，是 placement/proposal absence，而不是 closure selector错误。
- `new_task09_instance03` 的 fallback 达到 `Close=true` 但 `In=false`；这是唯一 placement loss，也是“fallback不是自动安全成功”的直接证据。

## 与 F2 相比的新信息

在两实验共享的 8 个 failure 上，F2 local oracle coverage 为6/8；F3 独立覆盖了F2未覆盖的 `new_task09_instance03`，所以模块化 proposal union 为7/8。F3 closure protocol没有取代F2：它没有恢复F2已覆盖的两个 drawer states，因此需要组合 proposal families，而不是宣布某一个方法普遍最好。

对 2 个新 task09 failures，1个能达到closure boundary并被local/fallback恢复，另1个完全没有placement proposal。因此 F3支持“articulation closure 是独立可执行模式”，同时也保留了proposal absence状态，为将来的fallback标定提供必要负例。

最重要的 representation 信息是：final success背后至少有独立的 `placement`、`articulation closure` 和 `handoff/fallback retention` 机制。一个 fallback 可以关门但破坏placement；一个 analytic proposal可以显著推进joint但未达到official success；一个 failure可以根本没有closure opportunity。这些都不能只用一个未分解标量解释原因。

## bug、smoke 与无效 run

- `experiments/EXP_F3_smoke/`：1个drawer boundary×6 methods的短接口验证，证明snapshot、predicate、joint和rollout链路可执行，不进入正式指标。
- 第一次正式输出完整保存在 `experiments/EXP_F3_invalid_contact_gate_and_gripper_sign/`。其中 aligned analytic 的960步全部停在 approach stage，而wrong control实际进入push，且`+1`被误当成open-gripper，导致13条placement loss。该run的policy outcomes是真实执行，但analytic mechanism comparison无效，因此整目录归档、不进入F3数字，也不删除。
- 修复依据来自保存trace而非outcome包装：LIBERO中`-1`才张爪；20步release后gripper约40 mm打开，随后lift保持`In`。新controller让两方向共享release/approach，并保证每个checkpoint都进入相反push stage。
- 正式 corrected run 后发现原runner漏写统一 `confirmed_failures.jsonl`；所有新failure `.npz`/snapshot已存在。`materialize_exp_f3_failure_manifest.py` 从这些保存证据重建manifest，并重新执行两次restoration验证。runner已同步修复，未来run会直接写manifest。

这些基础设施问题均在F3内解决，消耗0个额外EXP ID；无效目录和diagnostic evidence均保留。

## 审计、泄漏与限制

`experiments/EXP_F3/audit.json` 检查：10 confirmed failures、10 boundary rollouts/1,307步、8个独立恢复的closure snapshots、48 intervention rollouts/7,883步。首次boundary、final predicates、joint progress、gripper trace、aligned/wrong shared prefix、method matrix和所有aggregate均一致。

所有 choices 的反事实执行仅用于离线 oracle/proposal诊断；尚无 learned online selector，也没有执行全部choices后伪装成selected-only。当前 cohort小，仅有task03/task09且属于development；不能据此声称跨任务、真实机器人或final confirmation泛化。

## 支持与不支持的 claim

支持：closure是与placement不同的可执行recovery mode；joint/handle-aligned control产生方向特异的物理后果；closure-specific policy为F2 proposal set提供1/8 shared-state marginal coverage；release/placement retention是deployment handoff不可省略的 consequence；fallback会产生独立的 harmful outcome。

部分支持：analytic skill在microwave 2/4成功并在8/8优于wrong joint progress，但drawer 0/4，因此不是通用closure solution。

不支持：proposal coverage已经充分普适、fallback已可靠校准、factorized predictor优于scalar、unseen-state selector、selected-only多轮闭环或最终系统完成。

## artifacts 与磁盘

- runner：`scripts/experiments/run_exp_f3_articulation_closure_skills.py`
- manifest repair：`scripts/experiments/materialize_exp_f3_failure_manifest.py`
- audit：`scripts/experiments/audit_exp_f3.py`
- protocol/failures：`experiments/EXP_F3/frozen_protocol.json`、`confirmed_failures.jsonl`、`failures/`
- boundaries：`boundary_attempts.jsonl`、`closure_checkpoints.jsonl`、`checkpoints/`、`boundary_rollouts/`
- outcomes：`rollouts/`、`outcomes.jsonl`、`coverage_matrix.csv`
- summary：`metrics.json`、`run_metadata.json`、`audit.json`

完成后磁盘剩余846 GB；未删除任何已有实验、报告、checkpoint、log或无效run。

## F3 结论

F3完成了Priority B中针对articulation bottleneck的实质推进：模块化proposal union在共享cohort达到7/8，并用严格方向对照证明了closure consequence的物理含义。与此同时，未到placement、closure不足、以及fallback关门但丢placement三类failure已经可以独立诊断。下一步不应继续调push幅度，也不应立即用final success训练几个heads；应先冻结与真实deployment handoff完全一致的factor labels和proposal contract。
