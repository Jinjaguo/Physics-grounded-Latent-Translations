# EXP_F7 Report：Goal-Conditioned Analytic Coverage and Factor Intervention

## 实验有效性

```text
new_control_formulation: true
new_mechanism_interventions: true
confirmed_failure_states: 12 full-state canonical snapshots
actual_interventions: 60
executed_simulator_steps: 16192
machine_verifiable_artifacts: experiments/EXP_F7/{frozen_protocol.json,factor_schema.json,confirmed_failures.jsonl,proposals.jsonl,proposal_inputs/,rollouts/,outcomes.jsonl,factor_matrix.csv,metrics.json,run_metadata.json,audit.json}
audit_passed: true
audit_discrepancies: []
experiment_id_valid: true
```

F7复用F6的12个已审计full-state confirmed failures，但实际执行了新的goal-conditioned controller、wrong-direction/target control、approach-only control、长程semantic baseline和fallback，共60个新outcomes。审计检查12个model-state snapshots、60个outcome-free proposal inputs、60个rollouts、16,192步phase accounting、8组articulated shared prefixes及全部连续factor重算，零discrepancy。

## 科学假设与方法边界

F6 local oracle coverage只有2/12，且proposal-boundary object/articulation labels全负。F7不训练selector，而检验：在共享物体placement前段后，用目标关节和handle几何构造的aligned controller是否比wrong-direction与approach-only controls产生更好的joint progress和handoff；同时检验object target、contact、stability、joint progress与handoff是否是独立且有决策意义的consequences。

前三种controller读取simulator target site、handle和joint state，因此明确标为`privileged proposal oracle`，不是最终可部署方法或selector输入。它回答控制上是否存在coverage以及factor是否有机制含义，不能被引用为视觉部署性能。

## Proposal与matched intervention

每个state执行五种choice：`aligned_goal_control`、`wrong_direction_or_target`、`approach_only_control`、`semantic_policy_baseline`、`fallback_disengage_replan`。proposal最多260步，未完成时统一20-step release、5-step lift与120-step base-policy continuation；fallback只做30-step safe disengagement后从当前world state replan，不reset环境。

对于task03/09，三种mechanism controllers在`object_goal=false`时使用完全相同的placement prompt、随机seed、5-step feedback和动作。只有`object_goal=true`后才分别aligned push、wrong push或到达handle后不push。8个articulated failure groups的shared prefix长度、actions和object trajectories经审计逐元素相同。task05的correct-target、front-compartment、hold-above prompts并没有形成同等严格的物理matched control，因此task05只用于coverage/harm分析，不用于宣称target factor的因果必要性。

## Factor定义

F7在查看formal outcomes前冻结并保存连续值：object相对target site的local coordinate和normalized margin、EEF到handle/object的距离、目标joint qpos和normalized closure。二值接口包括：target margin改善至少0.10、官方`In`、最小距离不超过6 cm、normalized joint closure改善至少0.10（task03/09 mask）、最后10个proposal steps中至少80%保持`In`、exact handoff compatibility，以及target/joint regression或执行异常构成的harm。

这些factor不等同于最终success：proposal结束后仍严格执行真实release/lift与base continuation，`handoff_compatible`只在proposal未直接成功时定义。

## 主要结果

| 指标 | F7 | F6短时基线 |
|---|---:|---:|
| local oracle coverage | 8/12 = 66.7% | 2/12 = 16.7% |
| aligned controller coverage | 7/12 = 58.3% | 不适用 |
| no-local states | 4/12 | 10/12 |
| fallback coverage | 1/12 = 8.3% | 0/12 |
| final successful outcomes | 26/60 | 4/72 |
| harm/worsening | 8/60 | 0/72 |

方法成功率：semantic baseline 8/12、aligned 7/12、approach-only 6/12、wrong-direction/target 4/12、fallback 1/12。成功local方法数分布为：4个state无local，1个仅1种，4个有3种，3个有4种。

F7与F6的budget和proposal family不同，因此50个百分点不能归因于analytic controller单一组件；F7内部semantic baseline本身也达到8/12，说明更长的feedback policy已有相同state-level union coverage。analytic controller的价值主要来自matched mechanism差异，而不是相对该长程semantic baseline新增state coverage。

## Matched机制证据

在8个articulated states上，aligned相对wrong-direction最终结果为3胜、0负、5平；相对approach-only为1胜、0负、7平。aligned相对wrong平均增加0.379 normalized joint progress。代表案例：

- task03 instance06：三方法共享前115步；aligned增加0.398 joint progress并由handoff救回，wrong保持object但0 progress且失败，approach 0 progress但base continuation救回。
- task09 instance06：aligned增加0.785并proposal-direct成功；wrong自身0 progress但handoff救回；approach 0 progress且最终失败。
- task03 instance07：aligned已有+0.547 progress并保留object，仍handoff失败；semantic baseline直接成功。这反证“joint progress足以代表recovery”。
- task03 instance08：placement未达到`In`，三controls从未分叉，却共同把空drawer推进约+0.833，object target margin恶化；这是明确的“关空fixture”failure，而不是closure selector失败。

## Factor的保留、弱化与删除

在24个articulated matched outcomes中：

| 条件 | final success |
|---|---:|
| object/stability不满足 | 0/9 |
| object/stability满足 | 8/15 |
| joint progress正 | 4/8 |
| joint progress负 | 4/16 |
| target progress未达阈值 | 0/3 |
| target progress达阈值 | 8/21 |
| harm=true | 0/3 |
| harm=false | 8/21 |

因此object placement/retention是明确的阶段必要条件；joint progress能够区分aligned与wrong并解释direct closure，但不是最终handoff必要条件，应保留为阶段门控/诊断而非全局success head；handoff必须独立建模，因为相同0 joint progress既有可救回也有失败。当前`contact_ready≤6cm`为3/8成功，而未满足为5/16，没有决策价值；该factor按结果删除，后续若使用接触必须换成持续力/接触或object-acquisition定义，不能继续保留一个无用head。

## Proposal、selector与fallback诊断

4个no-local states全部来自F6新cohort：task03 instance08、task05 instance08、task09 instance08/09。它们主要未完成object acquisition/placement，因此仍是proposal coverage问题。已有有效proposal的8 states中不存在本次selector错误，因为F7是offline evaluator-only matrix。fallback只成功1/12，不能作为可靠兜底；同时它从当前world state执行，没有把home pose伪装成world reset。

8个harm中，5个来自task03 instance08各方法共同造成的巨大target regression，3个来自task05 instance08的aligned/approach/semantic target regression；全部发生在新states且最终失败。该结果说明harm factor有独立用途：它揭示某些长程proposal虽无异常/碰撞终止，仍物理上把任务物体带离目标。

## Bug、smoke与限制

- 第一次smoke在import阶段因命令环境少了`scripts/dynamics`而退出，未创建实验目录、未执行干预；修正脚本Usage和运行环境后原名重跑。
- `EXP_F7_smoke/`执行1个task03 state、5个完整outcomes，用于验证shared prefix与factor trace，不进入formal统计。
- 在运行前发现task05 masked joint使用NaN会被通用finite判据误报；修改为只在task05跳过明确masked joint字段，其他连续数组仍全部检查。
- task05 wrong-target/hold-above prompts常仍完成官方back-compartment目标，说明语言policy没有遵循负控制；不能把这些case作为task05 factor因果证据。

## Artifacts与磁盘

Runner为`scripts/experiments/run_exp_f7_goal_conditioned_oracle.py`，独立审计为`scripts/experiments/audit_exp_f7.py`。所有decision-time inputs仅含failure observation、method/prompt、privileged标记与first action prefix，没有future outcome。完成后磁盘剩余846 GB，未删除历史数据或artifacts。

## F7结论

F7首次用真实matched interventions证明factor interface可以参与控制结构：只有object factor满足后才进入closure，aligned joint consequence优于wrong control，而handoff又能独立改变最终结果。这不是把scalar拆成heads。但coverage仍只有8/12，且最主要失败是进入placement前没有可靠获得物体；contact-ready定义被数据否定。下一步应新增`object acquisition`阶段与factor-gated state machine，在未见confirmed failures上检验“先获得物体、再place、再close”是否提高coverage，同时保留harm与fallback诊断。
