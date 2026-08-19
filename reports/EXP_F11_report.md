# EXP_F11 Report：Outcome-Supervised Proposal-Set Viability and Structured Ranking

## 实验有效性

```text
development_full_state_groups: 30 (EXP_F8/F9/F10)
development_matched_outcomes: 120 (90 local + 30 fallback)
new_confirmed_full_state_failures: 12 (6 moderate + 6 severe)
pre-outcome_decisions: 12 x 5 selectors
actual_selected-only_rollouts: 60
strictly_post-selection_oracle_rollouts: 48
selected/oracle_steps: 14227 / 15540
machine-verifiable_artifacts: experiments/EXP_F11/{frozen_protocol.json,training_manifest.jsonl,training_metrics.json,state_oof.csv,proposal_oof.csv,checkpoints/,confirmed_failures.jsonl,restore_cross_env.jsonl,decision_inputs/,decision_manifest.jsonl,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
audit_passed: true
audit_discrepancies: []
experiment_id_valid: true
```

有效formal cohort为LIBERO official instances 23/24。所有12个events在decision前满足`env.check_success()==false`，完整full model state可跨env零误差restore。模型、representation、OOF threshold和全部choices在任何F11 outcome前冻结；60条selected-only全部完成后才运行48条oracle。

## 研究问题与数据改变

F10用EEF-object距离派生support label，但出现“几何近而三个local proposals仍全部失败”的state。F11把state-level target改成真正需要决策的量：对冻结的三local proposal set，`proposal_set_viable = any(local final_success)`。它只由历史真实interventions监督，不再拿距离充当ground truth。

为了保持proposal语义和budget一致，training只使用F8–F10中共同的`ungated_place_close`、`recovery_prompt_feedback`、`semantic_full_task`和fallback；F6/F7不同method family没有混入。30个unique failure groups包含17个viable和13个no-local。每个state/method只有一条真实outcome，selected与oracle重复没有重复计权。

## Representation与matched models

本机已有官方torchvision ResNet50权重。F11保存了exact encoder state，并从layer4 2D feature map做4×4 spatial pooling和32组channel pooling，双相机得到1024维visual feature，再拼8维robot state和task one-hot。与F10 16×16 raw RGB grid做相同5-fold full-state grouped OOF，representation只按development AUROC/Brier选择。

proposal-level输入再加入method one-hot、10×7 first-action prefix和valid length。五个ExtraTrees heads参数规模/输入一致：object readiness、safe/no-regression、direct completion、masked handoff compatibility以及matched scalar final success。structured score为`safe × max(direct, object × handoff)`。state viability gate由独立model输出，阈值0.635仅由development grouped OOF balanced accuracy冻结。

## Development OOF结果

| state representation | viability AUROC | Brier |
|---|---:|---:|
| F10 raw spatial grid | 0.466 | 0.290 |
| ResNet50 spatial feature | **0.760** | **0.207** |

冻结threshold下balanced accuracy为0.738，development false-support 3、false-fallback 5。ResNet feature相对raw grid是明确进步，但OOF错误率仍不支持“可靠viability estimator”的强claim。

| proposal target | positives / 90 | AUROC | Brier |
|---|---:|---:|---:|
| object readiness | 30 | 0.967 | 0.073 |
| safe/no regression | 83 | 0.807 | 0.091 |
| direct completion | 40 | 0.801 | 0.183 |
| masked handoff | 5 | 0.672 | 0.129 |
| matched scalar final | 45 | 0.722 | 0.230 |

与F9相比，safe head从AUROC 0.405提升到0.807，object/direct也具有独立监督意义；handoff仍只有5个正例，证据弱。task05上object head常输出接近0但direct head足以解释成功，说明object factor不是所有task的universal prerequisite，后续必须明确applicability mask。

development viable states的OOF ranking为factor 17/17、proposal prior 17/17、scalar 16/17。这提示factor可能有价值，但prior也饱和，必须看held-out selected-only，不能用OOF宣称优势。

## Formal oracle coverage与viability

有效formal恰好6/12有local recovery、6/12无local：6个moderate均至少一个local成功，6个经过confirmed-failure validation的severe均三local失败。

Outcome-supervised viability保留5/6 moderate、拒绝6/6 severe。唯一false-fallback是task09 instance23 moderate：概率0.537低于冻结0.635，但oracle中recovery-feedback与semantic都成功。因此fallback precision为6/7=0.857、recall为1.0、unsafe false-local为0。

F10 geometry visual ablation在本cohort保留6/6 moderate并拒绝6/6 severe，precision/recall均1.0。F11因此否定“outcome supervision已经普遍优于geometry”的强hypothesis。Outcome label在F11前一套无效cohort曾正确识别near-but-no-local task09 state，但在有效23/24 cohort损失一个recoverable state；30个训练groups还不足以稳定泛化。

## Selected-only结果

| selector | success | fallback | harm | fallback precision / recall | unsafe false-local | viable ranking miss |
|---|---:|---:|---:|---:|---:|---:|
| factor + outcome viability | 4/12 | 7 | 0 | 0.857 / 1.000 | 0 | 1 |
| scalar + outcome viability | 4/12 | 7 | 0 | 0.857 / 1.000 | 0 | 1 |
| viability-only + prior | 4/12 | 7 | 0 | 0.857 / 1.000 | 0 | 1 |
| scalar, no viability | **5/12** | 0 | 0 | 未定义 / 0.000 | 6 | 1 |
| scalar + F10 geometry visual | **5/12** | 6 | 0 | 1.000 / 1.000 | 0 | 1 |

Outcome gate的零unsafe false-local是真实fallback calibration改进，但本cohort无gate所选local恰好也没有触发harm，所以不能声称F11降低了harm。它相对geometry少1次task success，原因是task09 instance23 false-fallback。

6个viable states中，task03两个和task05两个都被factor/scalar正确恢复；task09 instance23被gate错误fallback；task09 instance24被正确放行，但所有factor、scalar、prior都选择`recovery_prompt_feedback`失败，唯一成功proposal是`semantic_full_task`。该state是proposal ranking error。factor预测recovery的direct=0.812、object=0.835，均高于semantic的0.783/0.803，错误具有明确factor attribution，而不是threshold问题。

formal中factor、scalar和prior choices的success完全相同，factor没有优势。虽然factor heads的development AUROC好于scalar，当前组合式接口没有把这些可解释预测转化为更好的held-out selection。因此可严谨支持的是“部分factors可独立监督并定位错误”，不能支持“factorized selector优于matched scalar”。

## Invalid run与constructor修正

第一次formal使用instances20/21，完整保存在`experiments/EXP_F11_invalid_presatisfied_instance20/`并有`INVALID_RUN.md`。其中task05 instance20 severe在物体位移后已经满足official goal，五个selected和四个oracle都`executed_steps=0/final_success=true`；它不是confirmed failure，整套run从F11 formal结论排除。

修正后的severe constructor从moderate snapshot按固定`away, perpendicular+, perpendicular−, toward`方向序列尝试30 cm displacement，只使用当前`env.check_success()`拒绝pre-satisfied candidate，不读取future recovery outcome。若四个方向都不能形成confirmed failure则在任何decision前报错。有效instances23/24的12个states全部通过，所有selected/oracle均为非零实际intervention。旧run未删除或覆盖。

`experiments/EXP_F11_smoke/`使用instance22验证训练、checkpoint与完整selected/oracle链；smoke outcome没有进入模型、threshold或formal state选择。

## 审计

`scripts/experiments/audit_exp_f11.py`重载保存的ResNet encoder和所有tree checkpoints，从48个decision archives重新提取spatial features，重算12个viability/geometry概率、180个proposal scores与全部choices，检查decision→selected→oracle时间顺序、first actions、failure policy state、trace长度与official success，并独立重算6/12 coverage和五个selected summaries。结果零discrepancy。

## F11结论

F11建立了以真实proposal-set outcomes定义的deployable viability factor，并证明预训练spatial representation显著优于raw grid；它在有效held-out cohort捕获全部6个no-local states且没有unsafe false-local。独立object/safe/direct factors有强development可监督性，能够把task09 ranking错误定位为对recovery proposal的direct/object过估计。

但F11没有证明outcome gate优于geometry，也没有证明factorized ranking优于scalar或简单prior。当前decision只看静态state和前10个policy actions，无法知道proposal实际开始执行后是否产生目标关节进展。F12应利用主线允许的`intervention → feedback → re-observe`：执行唯一selected recovery的短prefix后读取真实factor feedback，再决定继续、切换semantic、handoff或fallback，并通过删除object/progress feedback的机制ablation检验factor interface是否真正改变失败。
