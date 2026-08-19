# EXP_F6 Report：Full-State Short-Horizon Discriminative Benchmark

## 实验有效性

```text
new_confirmed_failure_data: 6
full_state_failure_count: 12
actual_recovery_interventions: 72
executed_simulator_steps: 12651
machine_verifiable_artifacts: experiments/EXP_F6/{frozen_protocol.json,confirmed_failures.jsonl,restore_cross_env.jsonl,proposals.jsonl,proposal_inputs/,rollouts/,outcomes.jsonl,factor_matrix.csv,metrics.json,run_metadata.json,audit.json}
audit_passed: true
audit_discrepancies: []
experiment_id_valid: true
```

F6不是排障或阈值搜索。它构造了6个新full-state confirmed failures，与F5的6个已审计full-state failures组成12-state canonical cohort；每个state真实执行6个不同proposal/fallback intervention。审计逐一检查12个model+integration snapshots、12个双fresh-env exact restores、72个outcome-free proposal inputs、72个rollout traces、12,651步phase accounting和72行factor matrix，全部一致。

## 科学问题与预注册解释

F5的200-step proposal、25-step handoff和160-step continuation使所有confirmation choices都成功，无法区分selector。F6检验：缩短为`80-step proposal → 20-step release + 5-step lift → 80-step base continuation`后，candidate outcome是否产生可选择的差异；同时保存执行前已知的首个policy action chunk，检验真实proposal content能否使articulation consequence比task/method prior更可预测。

支持模式是：存在足够多的candidate disagreement与articulation正负例，并且加入state/action content改善分组预测。反对模式包括：所有方法仍饱和、proposal coverage崩溃，或articulation label退化而无法训练。实际结果属于后两者：最终成功不再饱和，但proposal-boundary factor全部退化。

## Cohort与状态恢复

- F5来源：task03 instance06/07、task05 instance04/05、task09 instance06/07，共6个。
- F6新生成：task03/task05/task09的official init instance08/09，共6个。
- 所有snapshot均包含`model.body_pos/body_quat`与integration/controller-observable state；legacy F1–F4 snapshot没有进入该cohort。
- 每个snapshot在两个不同seed的fresh environment中恢复。integration、model body position/quaternion、8维policy state、双相机模型features和goal atoms的误差全部为0。
- confirmed failure由外部实验构造提供；没有训练或评估 impending-failure detector。

## Proposal与执行协议

每个state离线oracle评估6个choice：`full_task_short`、`fixed_phase_short`、`recovery_prompt_short`、`retrieval_recovery_short`、`closure_specific_short`和`fallback_short`。前5个local proposals最多80步、每5步重观察和replan；fallback执行20步release和10步lift，不重置world。未在proposal期成功时，统一执行25步disengagement及最多80步原始task base-policy continuation。

所有candidate在任何outcome生成前保存decision-time输入：8维policy state、两幅图像、prompt/stage，以及固定`10×7` raw first-action prefix与有效长度。审计验证每个learned/retrieval proposal实际执行的首5步与该prefix严格相同，fallback首10步相同；输入文件没有post-action state、reward、success或其他candidate outcome。

## 主要定量结果

| 指标 | 结果 |
|---|---:|
| confirmed failures / new failures | 12 / 6 |
| actual outcomes / simulator steps | 72 / 12,651 |
| local oracle coverage | 2/12 = 16.7% |
| no-local states | 10/12 |
| discriminative states | 2/12 |
| proposal-direct success | 0/72 |
| final success | 4/72 = 5.6% |
| handoff/continuation rescues | 4 |
| fallback success | 0/12 |
| harm/regression/exception | 0/72 |
| new-state success | 0/36 |

按方法，`closure_specific_short`为2/12，`full_task_short`和`fixed_phase_short`各1/12，`recovery_prompt_short`、`retrieval_recovery_short`与fallback均为0/12。按任务，task03为1/24、task05为0/24、task09为3/24。每state的local成功数分布是：10个state为0，1个为1，1个为3；没有所有local proposals都成功的state。

## Factor结果与机制案例

72个outcomes中，proposal-boundary `object_goal`为0；48个articulation-applicable outcomes中`articulation_goal`也为0。因此task/method、state+method及state+method+first-actions三套LOGO诊断都只有48个负例，AUROC未定义，恒负Brier为0。它们不是“完美模型”，而是标签退化，不能支持任何representation claim。

4个最终成功均发生在proposal并未满足官方factor后：

- task03 instance07：只有`closure_specific_short`成功；80步proposal、25步handoff后，base policy再执行53步完成。
- task09 instance06：`full_task_short`、`fixed_phase_short`、`closure_specific_short`分别在69、64、65步continuation后完成；同state另两种local proposal和fallback失败。

这两个案例证明proposal的历史会改变后续可完成性，也支持handoff consequence必须按真实continuation定义；但当前proposal-boundary官方predicate没有表达这种差异。10/12 no-local和6个新state的0/36则把主要失败定位为proposal coverage/cross-instance generalization，而不是selector没有找到已存在的proposal。

## Hypothesis判断

支持：short-horizon protocol消除了F5的全成功饱和，并生成2个真正有selection差异的state；同一snapshot上的不同proposal可导致不同handoff outcome；handoff不能被proposal-direct success替代。

否定：已知first-action content足以改善articulation prediction；当前官方`closed` predicate可直接作为短时独立articulation factor；现有prompt/retrieval proposal families在unseen instances上有足够coverage；fallback disengage+80-step replan能够可靠救回这些states。

尚未支持：factorized selector优于scalar/direct selector、可靠fallback calibration、多轮selected-only recovery loop、跨任务proposal generalization、richer-future prediction不必要。

## Bug、无效run与影响

- `EXP_F6_smoke/`：发现retrieval action chunk长度不同，不能直接stack；改为固定10×7 padding并保存`valid_action_count`。
- `EXP_F6_smoke2/`：单failure smoke进入leave-one-group-out时训练集为空；诊断现在对单组明确输出`insufficient_groups`。
- `EXP_F6_smoke3/`：修复后的接口smoke，完整执行6个outcomes，不进入正式统计。
- 第一次正式artifact audit错误地把10个保存动作与learned methods的前10个trace动作比较，忽略了每5步replan。逐元素检查确认首5步完全相等；audit判据修正为learned/retrieval比首5步、fallback比首10步，最终零discrepancy。实验数据没有因此修改或重跑。

上述目录均保留。它们不占EXP ID，也不支撑F6科学结论。

## Artifacts、代码与磁盘

Runner为`scripts/experiments/run_exp_f6_short_horizon_benchmark.py`，独立审计为`scripts/experiments/audit_exp_f6.py`。完整配置和code state见`experiments/EXP_F6/frozen_protocol.json`与`run_metadata.json`；逐case证据见`outcomes.jsonl`和`rollouts/*.npz`。完成后磁盘剩余846 GB，未删除任何研究artifact。

## F6结论

F6有效地把F5的selector饱和问题转化为可诊断结果，但也表明当前系统还不应继续训练selector：local oracle coverage仅16.7%，且短时object/articulation labels完全退化。下一步必须改proposal generator并重新定义可独立监督、能对应真实handoff机制的中间consequences；如果这些物理factor在matched interventions中也不改变结果，就应删除它们，而不是保留无意义的classifier heads。
