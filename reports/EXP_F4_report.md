# EXP_F4 Report：Deployment-Consistent Consequence Interface

> **F5勘误：** F4 matched legacy snapshots未保存LIBERO placement sampler写入的`model.body_pos/body_quat`。每个failure的六choice仍在同一restored variant上执行，factor/outcome相对关系有效；但不能声称fresh env逐像素等于最初failure-generation world。详见`reports/F1_F4_SNAPSHOT_MODEL_STATE_ERRATUM.md`。

## 实验有效性

```text
new_data_generated: true
new_model_trained: false
new_intervention_executed: true
new_deployment_consistent_evaluation: true
machine_verifiable_artifacts: experiments/EXP_F4/{frozen_protocol.json,confirmed_failures.jsonl,proposals.jsonl,failures/,rollouts/,outcomes.jsonl,factor_schema.json,factor_matrix.csv,metrics.json,run_metadata.json,audit.json}
experiment_id_valid: true
```

F4 在16个confirmed failures上执行6种choices，共96条真实interventions、24,087个控制步。4个failure来自新base-policy execution；其余12个来自audited F1/F3 snapshots。独立审计恢复全部16个failure state，逐条核对四个causal boundaries、phase counts、goal-atom traces、factor masks和aggregate，`passed=true`、`discrepancies=[]`。

## 为什么 F4 紧接 F3

F3已经把“未到placement”“已placement但未closure”“closure后丢placement”分开，但不同实验仍使用不同proposal/handoff时长。F4不训练selector，而是先冻结所有候选共享的deployment contract，确保模型将来预测的不是一个与online execution不一致的离线标签。

F4检验的假设是：在相同 `proposal → reobserve → disengage → base-policy continuation` 因果序列下，object progress、articulation completion、harm/regression和handoff compatibility会独立变化，因此是有物理/任务含义的interface，而不仅是多个任意heads。

## 冻结 causal contract

每个choice从同一完整snapshot独立恢复：

```text
confirmed failure / pending nominal queue cleared
→ execute one proposal, at most 200 steps
→ save real post-proposal images, policy state, integration state and goal atoms
→ if not complete: 20-step open-gripper settle + 5-step lift
→ save real post-handoff state
→ same fresh pi0.5 base policy, at most 160 steps
→ save final real state and official success
```

proposal提前成功则不伪造handoff。所有current/future task predicates只用于保存label和evaluation，不进入candidate selection；全部choices的执行是离线oracle/data generation，没有伪装成online selected-only。

## 数据 provenance

- F3 matched：10个task03/task09 failures；
- F1 matched：2个task05 book-placement failures；
- F4 new：task03 init 4/5与task05 init 2/3，共4个新base-policy executions和新5 cm displacement failures；
- 16/16 snapshots恢复误差为0；4个新failure的requested/realized displacement一致，重复restoration通过；
- task03/task09提供articulation mask，task05没有Close goal，不能把“不适用”当负例；
- 当前仍是development data，不是最终confirmation。

## proposal set

1. `full_task_feedback`：200-step full-task π0.5；
2. `fixed_semantic_proposal`：articulated tasks固定80-step placement后切closure；task05持续object-goal；
3. `recovery_prompt_feedback`：显式“recover current object, then task”；
4. `retrieval_then_recovery`：40-step train-success adapted retrieval后切recovery prompt；
5. `analytic_composite`：articulated tasks为80-step placement后analytic closure；task05在80步后切recovery prompt；
6. `fallback_disengage_replan`：20-step open settle +10-step lift，不reset world，随后进入同一standardized handoff/continuation。

local choice包含proposal和其后标准continuation的完整deployment outcome；proposal自身是否完成与handoff是否救回分别报告。

## factor 定义与实际保留/删除

### 保留进入 F5 的最小factor

- `proposal_object_goal`：post-proposal真实状态是否满足任务的`In` goal atom；56/96为正；
- `proposal_articulation_goal`：只在task03/task09适用，post-proposal `Close`；18/72为正；
- `handoff_compatible`：proposal未直接成功时，从真实post-handoff state运行相同base policy能否最终成功；23/61为正。

### 不作为 F5 learned head

- `object_retained`：56/56适用outcomes均为正；
- `articulation_retained`：18/18均为正。

这两个label说明修正后的统一open-gripper protocol有效，但没有负例，无法支持独立预测或selection claim。F5删除这两个heads；仍保留为运行时审计指标。

- `harm_or_worsening`：2/96为正，都是task09 instance04的真实goal-atom transient regression。它有distinct semantics，但样本太少，F5仅作为约束/evaluation，不声称已能学习calibrated harm head。

`final_success`为58/96，只作为matched scalar baseline和最终结果，不冒充mechanism factor。

## 定量结果

| metric | result |
|---|---:|
| confirmed failures | 16（4 new） |
| outcomes | 96 |
| control steps | 24,087 |
| proposal-direct success | 35/96 |
| continuation rescues | 23/61 applicable |
| final successes | 58/96 |
| oracle local coverage after exact contract | 16/16 = 1.000 |
| fallback final success | 4/16 = 0.250 |
| scalar-collision failures | 13/16 |
| supported exception/nonfinite outcomes | 0/96 |

### method final success

| method | success / 16 |
|---|---:|
| fixed semantic | 14/16 |
| full task | 13/16 |
| recovery prompt | 13/16 |
| retrieval→recovery | 10/16 |
| analytic composite | 4/16 |
| fallback disengage/replan | 4/16 |

task breakdown：task03共36 outcomes，proposal-direct 16、final 20、handoff rescue 4；task05共24 outcomes，proposal-direct 19、final 23、handoff rescue 4；task09共36 outcomes，proposal-direct 0、final 15、handoff rescue 15。task09清楚说明post-proposal factor与handoff factor不能合并：200步内没有proposal完成完整任务，但15条从真实handoff state继续后成功。

## 为什么 factorization 不是多个heads

articulated outcomes的 `(object, articulation, final)` joint counts为：

```text
(0,0,0): 31
(1,0,1): 17
(1,1,1): 16
(1,0,0): 4
(0,1,0): 2
(0,0,1): 2
```

这不是一个scalar拆写：

- `matched_task09_instance00`：full-task proposal为`object=1, close=0`，handoff后成功；analytic为`object=0, close=1`，即关闭空microwave，最终失败；其他失败choice为`0,0`。相同final-failure scalar包含“关错空机构”和“完全无进展”。
- `new_task09_instance03`：full-task以`object=1, close=0`进入handoff后成功；retrieval以`0,0`进入handoff后也成功。相同final-success scalar隐藏了handoff起点和恢复机制差异。
- task05四个states中，强local proposal直接完成object goal；fallback在post-proposal仍为0，但4/4由base policy从真实world state救回。相同final success需要不同控制语义。
- task09 instance04的full-task与analytic都最终失败，但分别产生In regression与Close regression；其他同state failure没有regression。harm不是final scalar的同义词。

16个states各有2–4种factor signatures；13/16在固定final outcome内仍有多个factor vectors。因而F4严谨支持“factor interface揭示独立机制”，但还没有证明预测factor比预测scalar更准或选择更好；那是F5的问题。

## proposal、selector 与 fallback decomposition

完整candidate contract的oracle local coverage为16/16，说明当前development cohort已有足够headroom进入selector研究；这不表示每个proposal都好，也不表示更广failure分布已覆盖。

fallback只有4/16且全部在task05成功；articulated tasks为0/12。因为16/16都有某个local success，F4没有“唯一正确choice是fallback”的state，不能据此训练/宣称fallback calibration。F5必须把fallback作为实际choice评估，但不能虚构fallback-positive/no-local labels。

## bug、smoke 与异常

- `experiments/EXP_F4_smoke/`：1个failure×6choices、短20-step proposal/continuation，仅验证四boundary、phase和mask写入，不进入正式metric。
- runner静态复核发现boundary logging最初调用了会canonicalize MuJoCo derived state的snapshot capture；正式运行前改成只读`capture_integration_state`，保证proposal从原snapshot逐字节状态开始。
- task05 analytic-composite初版与fixed prompt重复；正式运行前改为80步后使用显式recovery prompt，避免伪造不同proposal family。
- 正式96条outcomes无exception、nonfinite、restore mismatch或missing phase。

这些都是F4内基础设施修复，未消耗新EXP ID。

## 审计与限制

`experiments/EXP_F4/audit.json`检查16 snapshots、96 rollouts、24,087 steps；从trace index重建post-proposal/post-handoff/final atoms、retention masks、regression/harm、handoff、coverage和scalar collision，所有值一致。

限制包括：仅3个LIBERO tasks、16个failure groups；outcomes不是96个独立states；harm正例过少；fallback没有no-local正确选择状态；尚未训练任何predictor，也没有unseen selected-only evaluation。

## 支持与不支持的 claim

支持：deployment-consistent consequence interface已定义且可机器重现；object progress、articulation与handoff label独立变化；factor vectors在13/16 states中提供scalar没有的机制区分；统一release使retention稳定。

否定/删除：object-retention和articulation-retention当前不值得作为learned heads。

暂不支持：harm predictor、factorized优于scalar、unseen selector、fallback calibration、selected-only闭环或richer-future不必要。

## artifacts 与磁盘

- runner：`scripts/experiments/run_exp_f4_consequence_interface.py`
- audit：`scripts/experiments/audit_exp_f4.py`
- artifacts：`experiments/EXP_F4/`下的protocol、failure/proposal manifests、96 NPZs、outcomes、schema、matrix、metrics、metadata与audit

完成后磁盘剩余846 GB；未删除任何研究证据。

## F4 结论

F4完成了Research North Star Priority C的核心基础：预测对象已经从含糊的“recovery success”变成三个有独立物理/控制含义、与部署因果链一致的最小factor。数据也迫使我们删除两个退化heads并限制harm claim。下一步可以开始真正的unseen-state consequence prediction与selected-only选择，而不是继续修改label定义。
