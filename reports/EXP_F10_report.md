# EXP_F10 Report：Deployable Visual Support Gate

## 实验有效性

```text
development_support_samples / parent_groups: 312 / 24
new_full_state_failures: 12 (6 moderate + 6 severe)
pre-outcome_decisions: 12 x 5 selectors
actual_selected-only_rollouts: 60
strictly_post-selection_oracle_rollouts: 48
selected/oracle_steps: 15633 / 15838
machine-verifiable_artifacts: experiments/EXP_F10/{frozen_protocol.json,support_dataset_manifest.jsonl,support_samples/,support_oof.csv,checkpoints/visual_support.pkl,support_training_metrics.json,confirmed_failures.jsonl,restore_cross_env.jsonl,decision_inputs/,decision_manifest.jsonl,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
audit_passed: true
audit_discrepancies: []
experiment_id_valid: true
```

F10实际生成了312个新的rendered support samples、12个新的confirmed-failure full states，并在冻结全部decision后执行60条selected-only interventions，再执行48条只用于诊断的oracle interventions。正式cohort使用从未参与development或smoke的LIBERO official instances 18/19。instance16用于被否定的Ridge smoke，instance17用于分类器smoke；两者都没有进入正式统计。

## 研究假设与方法

F9已经证明显式local-support gate能避免在明显out-of-local states上盲目执行recovery，但该gate直接读取simulator中的EEF-object距离。F10检验：confirmed failure当前的双相机图像与robot state能否形成一个可部署、独立监督的local support factor，并在不使用future outcome的情况下替代privileged distance。

development dataset来自F6的12个、F8的6个和F9的6个moderate parent snapshots。每个parent在同一个restore state上渲染base以及沿away/toward/perpendicular方向10/20/30 cm的12个变体，共13 samples；所有同parent variants严格留在同一个OOF fold。输入是16×16双相机RGB spatial grids、8维policy state和task one-hot，共1547维。physical support label由F9真实recoverable/no-local outcomes之间的task-wise距离中点生成。最终gate是400-tree ExtraTreesClassifier；概率阈值0.685只由312个grouped OOF predictions的balanced accuracy冻结。Ridge距离预测只作为诊断，不参与正式选择。

五个实际selector共享同一proposal set：factorized + visual support、scalar + visual support、scalar + privileged support、scalar without support、visual-support-only + task/proposal prior。所有selector在12个state上的choice先写入`decision_manifest.jsonl`，然后分别只执行被选中的branch；oracle始终在全部selected完成后运行。

## Development OOF结果

| 模型/目标 | 指标 | 结果 |
|---|---:|---:|
| Ridge EEF-object distance | grouped OOF MAE | 0.378 m |
| Ridge distance ranking support | AUROC | 0.612 |
| ExtraTrees direct support | AUROC | 0.875 |
| ExtraTrees direct support | balanced accuracy | 0.808 |
| ExtraTrees @ frozen 0.685 | false-support / false-fallback | 14 / 52 |

距离回归误差大于任务workspace尺度，不能作为定量距离估计器。直接分类明显优于先回归距离再阈值化，但OOF仍偏保守，有52个false-fallback。因此F10的正面结论只针对正式selected-only结果，不能把0.875 AUROC包装成可靠的通用几何估计。

## 正式proposal coverage与support error

事后oracle显示5/12 states至少有一个成功local proposal，7/12没有。构造标签与真实coverage并不完全相同：6个severe全为no-local，但task09 instance19 moderate的三种local也全部失败。

视觉gate保留5/6 moderate并拒绝6/6 severe。按真实oracle coverage而不是构造机制计算，它有：

- 6个true fallback；
- 1个false-fallback：task05 instance18 moderate，视觉概率0.641低于0.685，但ungated与recovery-feedback都真实成功；
- 1个false-support：task09 instance19 moderate，视觉概率0.950且距离仅0.270 m，但三种local全部失败；
- fallback precision/recall均为6/7 = 0.857。

privileged distance同样把task09 instance19判为supported，所以F10揭示的关键问题不只是视觉误差：EEF-object几何接近是local viability的必要线索，却不是“当前proposal set中存在有效recovery”的充分条件。proposal-set coverage必须由真实outcome label监督，不能继续拿距离label替代。

## Selected-only结果

| selector | success | fallback | harm | fallback precision / recall | unsafe false-local |
|---|---:|---:|---:|---:|---:|
| factorized + visual support | 3/12 | 7 | 0 | 0.857 / 0.857 | 1 |
| scalar + visual support | **4/12** | 7 | 0 | 0.857 / 0.857 | 1 |
| scalar + privileged support | **4/12** | 6 | 0 | 1.000 / 0.857 | 1 |
| scalar, no support | **4/12** | 0 | 2 | 未定义 / 0.000 | 7 |
| visual support-only + prior | 3/12 | 7 | 0 | 0.857 / 0.857 | 1 |

视觉gate与无support scalar的task success同为4/12，但将harm从2降为0、unsafe false-local从7降为1。两个harm分别出现在task03 instance18 severe与task09 instance18 severe；在完全相同的confirmed state上，visual gate选择fallback且不产生harm。由此支持的结论是“可部署视觉support factor改善harmful-recovery avoidance和fallback calibration”，而不是“提高总体task success”。

fallback在正式cohort所有selected/oracle traces中都无harm，但短budget内没有完成任务。它表示safe disengage/replan handoff，不表示world reset或任务恢复。第一次Ridge smoke的instance16上fallback本身出现target regression，进一步说明不能预设任何fallback实现天然安全；正式结论只覆盖当前cohort和当前fallback controller。

## Proposal generator、selector与handoff分解

- task03 instance18：oracle有recovery-feedback和semantic两个成功proposals；factorized与support-only选ungated失败，scalar选semantic成功。这是selector error，不是support或coverage error。
- task05 instance18：oracle有ungated和recovery-feedback两个成功proposals；视觉gate错误fallback，同时privileged scalar选semantic也失败。该state同时暴露visual support error和scalar ranking error。
- task09 instance19：三个local proposals全失败，属于proposal coverage failure；视觉与privileged support都误判为local-supported，但所选recovery未产生harm。
- 其余4个recoverable states中，visual scalar全部选到成功proposal；6个severe均被visual gate拒绝。

factorized + visual只有3次成功，仍低于matched scalar + visual的4次，且与support-only相同。F10因此再次否定“当前factor heads整体优于scalar”的强hypothesis。已有factorization的有限机制证据仍来自F7的factor deletion和F9的handoff ablation；F10没有新增总体优势。新的独立support factor有清楚物理含义和harm/fallback作用，但它尚未准确表达proposal-set viability。

## 无效run、bug与修正

- `experiments/EXP_F10_smoke/`（instance16）训练Ridge距离模型并用90% residual upper bound控制gate。grouped OOF MAE 0.378 m、support accuracy 0.353、202/312 false-fallback；新moderate也被错误fallback。该方法按数据否定，完整保留。
- ExtraTrees/RandomForest距离回归的离线诊断仍有约0.17 m MAE，upper-bound gate仍产生202个false-fallback，没有进入实验decision。
- 第一次classifier启动把字符串parent ID错误转为整数seed，在训练前退出；不完整目录`experiments/EXP_F10_smoke_classifier/`保留，不占EXP ID。修复为排序group序号seed。
- `experiments/EXP_F10_smoke_classifier_v2/`（instance17）验证direct classifier能保留moderate、拒绝oracle-confirmed no-local severe；这才触发instances18/19 formal run。
- smoke outcomes没有用于正式probability threshold、模型训练或formal state选择。

## 审计

`scripts/experiments/audit_exp_f10.py`重新读取312个support archives并验证24×13 grouping，重放checkpoint得到12个visual probabilities，从F9 factor/scalar checkpoints重算60组proposal scores和全部choices，检查decision→selected→oracle文件时间顺序、selected first actions、failure state、trace长度、official success以及oracle矩阵，并独立重算coverage和所有selected summaries。结果零discrepancy。

## F10结论

F10把F9的privileged support oracle替换成了真正只用decision-time图像与robot state的模块，并用selected-only干预证明该模块可以消除本cohort中2次harmful local execution。这个support factor不是把final scalar拆成另一个head：它有独立的当前状态监督、独立gate作用，并改变fallback decision而不改变proposal ranking。

但F10也否定了两个更强结论。第一，当前视觉模型还会牺牲可恢复state；第二，几何support本身无法覆盖“近但当前proposal set仍无解”的情况。完整factorized selector仍没有超过matched scalar。下一实验必须把support label改成由真实local intervention outcomes定义的proposal-set viability，并用更强的spatial visual representation联合解决support与proposal ranking，而不能继续微调0.685阈值。
