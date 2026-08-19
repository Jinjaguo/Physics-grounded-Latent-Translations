# EXP_F5 Report：Frozen Factor Predictor and Selected-Only Unseen Evaluation

## 实验有效性

```text
new_data_generated: true
new_model_trained: true
new_intervention_executed: true
selected_only_before_oracle: true
machine_verifiable_artifacts: experiments/EXP_F5/{frozen_protocol.json,training_split.json,feature_schema.json,checkpoints/,oof_predictions.csv,confirmed_failures.jsonl,decision_inputs/,decision_manifest.jsonl,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
experiment_id_valid: true
```

F5只用F4的16个failure groups/96 outcomes训练模型；模型、feature transform、OOF thresholds和decision rule全部冻结后，才生成task03/task05/task09各2个、共6个新confirmed failures。每个state先在即将执行的同一个fresh env中保存decision input和唯一choice，再执行1条selected-only branch；随后才从原snapshot离线执行6-choice oracle matrix。正式数据包含6条selected rollouts（1,121步）与36条oracle rollouts（7,609步）。独立审计重算全部prediction、choice、factor/outcome metric与rollout replay，`passed=true`、`discrepancies=[]`。

## 模型与输入边界

输入仅含decision-time可得的8维policy state、双相机固定4×4 RGB block means、task one-hot、proposal method one-hot与articulation mask。没有post-proposal state、执行后action、goal outcome、其他candidate outcome或future continuation结果。

factorized与scalar均使用114维输入、32-unit tanh shared encoder；参数量分别3,779与3,713，差1.8%。固定600 epochs AdamW；16-fold leave-one-failure-out仅在F4上完成。三个masked factors是`proposal_object_goal`、task-masked `proposal_articulation_goal`和proposal-incomplete-masked `handoff_compatible`；retention heads已删除，harm仅作evaluation。

## OOF development结果

| factor head | n / positive | AUROC | AP | Brier |
|---|---:|---:|---:|---:|
| object | 96 / 56 | 0.837 | 0.879 | 0.203 |
| articulation | 72 / 18 | 0.800 | 0.618 | 0.180 |
| handoff | 61 / 23 | 0.666 | 0.451 | 0.290 |

| final predictor | AUROC | AP | Brier | log loss |
|---|---:|---:|---:|---:|
| factorized | 0.814 | 0.832 | 0.180 | 1.480 |
| matched scalar | 0.815 | 0.839 | 0.200 | 1.357 |
| object-only | 0.771 | 0.790 | 0.244 | 1.796 |
| no-articulation | 0.782 | 0.793 | 0.193 | 1.840 |
| no-handoff | 0.819 | 0.871 | 0.270 | 2.466 |
| task/method prior | 0.826 | 0.836 | **0.141** | **0.744** |

development不支持factorized全面胜过scalar：Brier更好，但AUROC/AP略低且log loss更差；简单prior校准最好。所有selector的OOF fallback threshold均为0，因为F4的16/16 states都有local success，无法校准abstention。

## Untouched confirmation prediction

| factor head | n / positive | AUROC | AP | Brier |
|---|---:|---:|---:|---:|
| object | 36 / 26 | **0.992** | **0.997** | **0.099** |
| articulation | 24 / 10 | **0.471** | 0.451 | **0.415** |
| handoff | 16 / 7 | **0.921** | 0.874 | **0.149** |

object与handoff在新states上有强signal；articulation接近随机且严重失准，否定当前coarse state+method representation能泛化预测closure consequence。

| final predictor | AUROC | AP | Brier | log loss |
|---|---:|---:|---:|---:|
| factorized | 0.934 | 0.969 | 0.098 | 0.543 |
| matched scalar | 0.938 | 0.977 | 0.155 | 0.937 |
| object-only | **0.996** | **0.999** | 0.127 | 0.619 |
| no-articulation | 0.971 | 0.990 | **0.065** | **0.362** |
| no-handoff | 0.922 | 0.974 | 0.412 | 3.973 |
| task/method prior | 0.912 | 0.939 | 0.073 | 0.549 |

factorized相对matched scalar有更好的Brier/log loss但AUROC略低。去掉handoff后Brier恶化4倍，支持handoff factor；去掉articulation后反而改善到0.065，明确否定当前articulation head的generalization benefit。prior仍很强，factorized没有全面胜过简单统计。

## Selected-only结果

outcome前冻结的factorized choices为：task03 instance06/07选fixed semantic；task05 instance04/05选full task；task09 instance06/07选recovery prompt。6/6 selected-only成功、0 harm、0 fallback；同方法oracle重放6/6 action/stage/goal traces一致，oracle local coverage也是6/6，regret为0。

但是scalar、object-only、no-articulation、no-handoff与prior各自选择的candidate也全部6/6成功。当前200+25+160 contract过于宽松，多个proposal都能被continuation救回。因此F5证明factor selector可执行，却不证明其selection优于ablation或prior。

## Snapshot/model-state bug与无效runs

F5发现旧`LiberoSnapshot`没有保存placement sampler写入的`model.body_pos/body_quat`。不同seed fresh env即使integration error为0，fixed-body placement可差1.76 cm，图像平均差可达9.4像素。snapshot schema现已保存/恢复这两个model arrays；跨seed验证integration、model arrays、双相机、EEF与gripper max error均为0。

- `EXP_F5_smoke/`：50-epoch、1-state短authority接口验证，不进入正式结果。
- `EXP_F5_invalid_missing_model_body_state/`：第一次正式run；decision与selected world不一致，audit失败，完整保留。
- `EXP_F5_invalid_decision_render_context/`：model state已修复，但decision在failure-generation render context读取、selected在另一fresh context执行；visual block features仍有0.002–0.010差异，audit失败，完整保留。
- 最终`EXP_F5`在selected执行的同一个fresh env内restore、保存`decision_inputs`、写decision，再执行selected；审计证明decision与selected integration、policy state与实际视觉features一致。

这些都在F5内解决，未消耗额外ID，也未删除无效证据。

## 对F1–F4的勘误影响

F1–F4 legacy snapshots没有model-state字段。同一实验内，每个failure的choices仍从同一个restored variant开始，所以真实intervention、method对照与相对coverage仍有效；但“fresh env完整等于最初failure-generation world”的表述过强，旧audit没有检查model arrays。F6将用F5以后full-state snapshots重建canonical benchmark。

## 支持、否定与限制

支持：decision-before-outcome和selected-only链成立；object/handoff在unseen states有强signal；factorized比matched scalar校准更好；删除handoff产生明确预测失败。

否定：当前articulation predictor泛化；factorized selected success优于scalar/ablations/prior；fallback calibration；宽松contract能区分selector。

尚不支持：repeated multi-cycle loop、no-local正确fallback、跨任务/真实机器人、richer future不必要或最终paper claim。

## Artifacts、审计与磁盘

runner为`scripts/experiments/run_exp_f5_factor_selector.py`，audit为`scripts/experiments/audit_exp_f5.py`，snapshot修复在`src/pglt/libero/snapshot.py`。审计检查96 OOF rows、6 full-state snapshots、6 decision inputs、6 selected rollouts与36 oracle rollouts，0 discrepancies。磁盘剩余846 GB。

## F5结论

F5建立了第一个有效unseen selected-only predictor→choice→execution链，并给handoff factor提供强机制证据；但未证明完整factor selector优于简单替代。下一步不能调hidden size或threshold，必须重建full-state canonical benchmark并缩短contract，使candidate outcome真正产生selection/fallback disagreement，同时保存decision-time proposal content以解决articulation head失败。
