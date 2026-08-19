# EXP_F8 Report：Factor-Gated Regrasp–Place–Close on Untouched Failures

## 实验有效性

```text
new_confirmed_failures: 6 (task03/05/09 official init 10/11)
new_factor_gated_control: true
actual_interventions: 30
executed_simulator_steps: 7384
machine_verifiable_artifacts: experiments/EXP_F8/{frozen_protocol.json,factor_schema.json,confirmed_failures.jsonl,restore_cross_env.jsonl,proposal_inputs/,rollouts/,outcomes.jsonl,factor_matrix.csv,metrics.json,run_metadata.json,audit.json}
audit_passed: true
audit_discrepancies: []
experiment_id_valid: true
```

F8只在F6/F7 cohort上定义controller与阈值，正式结果来自6个新instance10/11 confirmed failures。每个state执行factor-gated、ungated、通用recovery、semantic full-task和fallback，共30个新outcomes。审计检查6个含model state的snapshot、6个双fresh-env zero-error restores、30个outcome-free inputs、30个rollouts、7,384步、factor masks与aggregate，全部一致。

## 科学假设与控制结构

F7的4个no-local states主要未先获得/放置物体。F8检验强制`object_acquired` gate能否避免关空fixture和target regression：gated proposal先执行显式pick；只有物体相对failure高度提升至少4 cm、EEF-object距离不超过14 cm且连续3步满足，才进入place；只有官方`In=true`后，task03/09才允许analytic closure。ungated ablation直接place，再由同一object factor决定是否进入closure。

`object_acquired`、`object_goal/retained`、object-masked joint progress、exact handoff和harm均在formal outcome前冻结。F7无效的瞬时`contact_ready≤6cm`已删除。所有动作由robot/controller执行，没有teleport或world reset。

## 主要结果

| 指标 | 结果 |
|---|---:|
| confirmed failures / outcomes | 6 / 30 |
| proposal-set local coverage | 6/6 |
| factor-gated success | 3/6 |
| ungated success | 5/6 |
| recovery-prompt success | 6/6 |
| semantic full-task success | 6/6 |
| fallback task success | 0/6 |
| harm/worsening | 0/30 |

gated相对ungated为1胜、3负、2平；两者的acquisition count均6/6、object-goal count均5/6，gated acquisition advantage=0、object-goal advantage=0、harm reduction=0。因此gate没有改善假设中的`acquire → place`转化，也没有减少本cohort中本来就没有出现的harm。

按phase，gated 6个proposal共1,636步，3个proposal-direct成功、0个handoff rescue；ungated共1,493步，3个direct加2个handoff rescue；recovery prompt与semantic分别1,184/1,261 proposal steps并全部直接成功。强制pick阶段不仅没有增加信息，还在部分state减少了后续closure/handoff可用的有效状态。

## 成功与失败案例

- task03 instance10/11：gated都acquire并达到`In`，但最终0/2；ungated、recovery和semantic均2/2成功。gate确认了一个已经会自然发生的事件，却延长路径并恶化最终handoff。
- task05 instance10/11：四种local methods都直接成功；gate只改变步数，没有决策收益。
- task09 instance10：gated成功而ungated虽acquire但未保持object goal，形成gated唯一一胜；然而recovery/semantic也成功，因此没有新增proposal-set coverage。
- task09 instance11：gated未达到object goal并失败，ungated/recovery/semantic成功，反向否定单例收益可泛化。

## Factor判断

30个outcomes中proposal acquisition 24、object goal 22、object retention 13、joint-progress applicable 14且正例14、official articulation goal 11、final success 20、handoff rescue 2、harm 0。joint progress只有在proposal-boundary object goal已经成立时才applicable，因此task03/09的关空fixture motion不会再成为虚假正例；这个mask是F8保留的有效interface修正。

但`object_acquired`作为强制gate被否定：所有四个local方法都是6/6 acquire，它没有区分最终choice；gated与ungated的object goal相同却final success不同，说明额外阶段历史/剩余可控性不能由这个bit表达。可保留连续lift/distance用于proposal诊断，不应作为最终系统中每次都必须先满足的factor/head。

本cohort harm全负，不能用F8证明harm predictor；F7的target-regression案例仍是现有机制证据。handoff只有ungated的2次rescue，继续支持proposal-boundary goal与真实continuation结果要分开。

## Proposal、selector与fallback诊断

新cohort local coverage为6/6，说明现在可以进入consequence prediction/selection阶段；但recovery和semantic都6/6，当前自然cohort对它们仍饱和，不能证明selector优于简单prior。F9需要构造预先定义的多failure-mechanism benchmark，同时包含可恢复与没有可信local recovery的states。

fallback在6个states上都未在120-step continuation内完成任务。这不等于fallback错误：safe disengagement本身无harm，但“是否正确选择fallback”必须依据local proposal是否可信以及安全交给replanner，不能继续把短期task success作为fallback唯一标签。

## Bug、smoke与审计说明

- `EXP_F8_smoke/`生成1个新task03 state并执行5个完整outcomes，用于验证gate真实触发和joint mask；不进入formal统计。
- 第一次artifact audit把handoff/base continuation期间的物体抬起也算成proposal acquisition，导致5个fallback假差异。runner的factor在proposal边界冻结是正确语义；审计改为只在`phase==0`的连续窗口重算后零discrepancy。实验数据未修改。
- 未引入analytic regrasp oracle，因为纯语言pick已在smoke和formal全部local methods上触发acquisition；问题不是gate无法执行，而是gate没有决策价值。

## Artifacts与磁盘

Runner为`scripts/experiments/run_exp_f8_factor_gated_regrasp.py`，审计为`scripts/experiments/audit_exp_f8.py`。所有choice先保存failure observation、prompt/method与first action prefix；没有future outcome leakage。完成后磁盘剩余846 GB，未删除任何历史artifact。

## F8结论

F8在未见states上把proposal-set coverage推进到6/6，但明确否定“强制object-acquisition gate普遍改善recovery”。有意义的factor set应收缩为：object goal/retention或连续progress、object-gated articulation progress、harm/worsening与deployment-consistent handoff；acquisition仅作proposal内部诊断。下一步应训练并冻结factorized consequence model和matched scalar/direct baselines，再在包含recoverable与no-trustworthy-local states的新benchmark上做decision-before-outcome、selected-only和post-selection oracle评估。
