# EXP_F9 Report：Mixed-State Factorized Risk Selector and Safe Fallback

## 实验有效性

```text
training_outcomes/groups: 162 / 18
new_full_state_failures: 12 (6 moderate + 6 severe)
pre_outcome_decisions: 12 x 8 selectors
actual_selected_only_rollouts: 48 (4 selectors x 12 states)
post_selection_oracle_rollouts: 48 (4 choices x 12 states)
selected/oracle_steps: 11363 / 14806
machine_verifiable_artifacts: experiments/EXP_F9/{frozen_protocol.json,training_manifest.json,feature_schema.json,checkpoints/,oof_predictions.csv,confirmed_failures.jsonl,restore_cross_env.jsonl,decision_inputs/,decision_manifest.jsonl,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
audit_passed: true
audit_discrepancies: []
experiment_id_valid: true
```

F9把F6–F8的162个真实outcomes按18个unique full-state failure分组训练；F6/F7同snapshot的rows始终在同一LOGO fold。模型、normalization、OOF thresholds、proposal set、severe construction和selectors都在confirmation前冻结。正式cohort为instance14/15：每个task各2个moderate failure，并从同snapshot把物体沿远离当前EEF的水平方向移动30 cm、settle 10步构造paired severe state，共12个。所有12个decisions先写完，随后分别执行factorized、scalar、scalar-with-support、support-only的唯一branch；全部48条selected完成后才跑48条oracle。

## Representation、模型与输入边界

输入是decision-time 8维robot state、双相机4×4 RGB block means、task/articulation mask、10维proposal语义描述、首个10×7 action chunk和冻结budgets，共190维。禁止post-action state、reward、success和其他candidate outcome。

factorized shared-encoder预测四个独立labels：proposal-boundary object readiness、no-harm/safe、proposal-direct completion、proposal incomplete时的masked handoff compatibility。matched scalar用同输入和hidden dimension直接预测final success。selector还包括no-harm、no-handoff、scalarized-factor、task/proposal prior。

F9 smoke发现所有learned OOF thresholds为0，不能识别30 cm distribution shift。高维state nearest-neighbor与target margin都无法在development区分no-local。最终增加一个显式physical support factor：task-wise development最大EEF-object距离加2 cm tolerance；超出时factorized及matched support ablations选择fallback。该距离由simulator state读取，明确是`privileged support oracle`，不是可部署视觉输入。

## OOF prediction

| target | n / positive | AUROC | AP | Brier |
|---|---:|---:|---:|---:|
| object readiness | 162 / 53 | 0.923 | 0.744 | 0.124 |
| safe/no-harm | 162 / 154 | 0.405 | 0.955 | 0.083 |
| proposal-direct | 162 / 35 | 0.890 | 0.705 | 0.140 |
| masked handoff | 127 / 15 | 0.602 | 0.205 | 0.195 |

factorized final的AUROC/Brier为0.765/0.221，matched scalar为0.722/0.296；factorized在development ranking/calibration较好。但safe head AUROC低于随机、handoff仅0.602且正例很少，不能由OOF alone主张可靠risk/handoff prediction。

## Mixed benchmark与oracle coverage

Oracle结果恰好6/12有local proposal、6/12无local：6个moderate states全部至少1个local成功，6个severe states的三种local全部失败；fallback在所有states都安全、无harm，但在短budget内不完成任务。严重机制因此提供了有效的fallback-correct condition，而不是通过事后挑选state得到。

task03 instance14 severe的三local都产生target-regression harm；其他5个severe local虽失败但不一定触发harm。因此“no viable local”不能由harm head替代。fallback correctness按oracle local absence与safe disengagement定义，不把home pose或短期任务成功当world reset。

## Selected-only结果

| selector | selected success | fallback | harm | oracle fallback precision / recall | unsafe false-local |
|---|---:|---:|---:|---:|---:|
| factorized + support | 4/12 | 7 | 0 | 0.857 / 1.000 | 0 |
| scalar, no support | 6/12 | 0 | 1 | 未定义 / 0.000 | 6 |
| scalar + same support | **5/12** | 7 | 0 | 0.857 / 1.000 | 0 |
| support-only + prior ranking | 4/12 | 7 | 0 | 0.857 / 1.000 | 0 |

无support scalar的6次success都来自recoverable states，但它在6个no-local states全部错误选择local，并在task03 instance14 severe产生1次harm。因此单看task success会掩盖严重fallback failure。

support-constrained methods全部捕获6个true fallback，但也把task05 instance14 moderate误判为fallback：该state EEF-object距离0.297 m，只比冻结阈值高约1 mm，oracle三local全成功。这给support precision=6/7并揭示max-distance calibration非常脆弱。

在6个recoverable states中，scalar+support成功5个，factorized+support成功4个。task09 instance15 moderate上factorized选semantic失败，而scalar+support选ungated成功；这是factor ranking error，不是support error。support-only也成功4个，说明当前factor heads没有提供总体增益。

## Mechanism ablations

post-selection oracle对冻结choices重算：factorized与no-harm均4次成功，表明当前safe head没有改变关键选择；删除handoff降到3次成功并增加到9次fallback，说明handoff factor至少在一个recoverable state提供决策价值；scalarized-factor与support-only同为4次。matched scalar+同support为5次，仍优于完整factorized。

因此F9提供了一条有限但真实的factor机制证据：handoff head的删除会造成可解释的额外fallback/成功损失；同时它否定更强结论——当前完整factorization并不优于matched scalar，safe head也没有selection价值。

## Smoke、无效run与修正

- `EXP_F9_smoke/`使用instance12 pair、100 epochs验证完整链。原始learned selectors对severe全部false-local并harm，证明OOF threshold退化。
- state-feature最近邻和initial target margin均无法在development区分no-local，按数据否定，没有加入formal。
- EEF-object support在instance12 smoke区分moderate/severe后加入；为避免confirmation tuning，instance12/13均只作development smoke，formal改用未见instance14/15。
- `EXP_F9_smoke_support/`使用instance13验证support-constrained selectors在severe选择fallback、scalar no-support产生harm；完整保留但不进入formal统计。
- severe construction由away-target改为更直接的away-current-EEF，formal protocol在outcome前冻结并记录。

## 审计与磁盘

`scripts/experiments/audit_exp_f9.py`在12个fresh env中重算privileged support，从两个checkpoints重算48组prediction与全部choices，检查48 decision inputs、48 selected traces、48 oracle traces及时间顺序；零discrepancy。完成后磁盘剩余846 GB，没有删除历史artifact。

## F9结论

F9首次建立了机器可验证的mixed recoverable/no-local selected-only benchmark，并证明显式fallback support约束能消除unsafe false-local；也证明fallback安全不等于短budget任务成功。但该support目前依赖privileged simulator distance并有一次边界false fallback，不能作为最终部署模块。更重要的是，factorized+support不如matched scalar+support，只有handoff head在ablation中显示有限价值。下一步要把当前状态support变成从图像估计的、带校准区间的可部署factor，并在相同support下继续公平比较factorized与scalar ranking。
