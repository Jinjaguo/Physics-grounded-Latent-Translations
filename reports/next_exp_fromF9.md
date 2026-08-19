# Next Experiment from EXP_F9：EXP_F10 Visual Reachability Support with Matched Selectors

## F9后的主线状态

F9 mixed benchmark有6个recoverable与6个no-local states。privileged EEF-object support实现fallback recall 1.0但precision 0.857；factorized+support在recoverable states成功4/6，scalar+同support成功5/6。主瓶颈现在是support factor不可部署/边界失准，以及factor ranking仍不如scalar，而不是proposal coverage。

## 新科学假设

> 当前状态的local reachability可以作为独立、可监督的物理factor从双相机图像与robot state估计；使用预测距离的calibrated upper bound而非硬max阈值，可保留moderate边界state并拒绝out-of-local-workspace severe states。若factorized与matched scalar共享同一visual support gate，二者差异才可归因于consequence ranking而非privileged fallback信息。

F10不预测impending failure；视觉模型只在confirmed failure后估计当前物体相对robot的local recovery support。

## Visual support dataset

- 使用F6–F9 development snapshots，在每个full-state failure附近预先定义多方向0/10/20/30 cm物体位移，restore、settle并render双相机；保存真实EEF-object距离与是否在task-wise local support域。
- instance14/15 formal outcomes可作为development历史，但不得用F10 confirmation instance16/17调阈值。
- 图像输入保留空间结构，至少使用16×16 RGB block grid或小CNN；与F9的4×4全局block features比较。
- split按parent snapshot与official instance分组，所有位移variants留在同fold，避免同场景泄漏。
- 输出distance regression、support probability与calibrated upper bound；阈值只由development physical tolerance和grouped calibration确定。

## Untouched selected-only cohort

新生成instance16/17的6个moderate+6个away-EEF severe states。所有decision和visual support predictions先冻结。实际执行：

1. factorized + visual support；
2. scalar + visual support；
3. scalar + privileged support（diagnostic upper bound）；
4. scalar without support；
5. visual-support-only + prior ranking。

随后才执行4-choice oracle。继续按fallback correctness、安全性、task success和selector regret分解。

## 必须回答

- visual support对moderate/severe的AUROC、距离MAE、coverage与calibration；
- fallback precision/recall和moderate false-fallback，尤其F9的1 mm边界问题是否减少；
- visual与privileged support性能差距；
- 在相同visual support下factorized是否仍不如scalar；
- support-only是否仍能fallback但不会正确rank proposals；
- false-local是visual factor错，还是ranking/threshold错。

若visual estimator不能跨instance泛化，不得把privileged support留在最终系统；F11应改用object-centric detector/segmentation或主动的post-failure perception，而不是调一个距离阈值。

## 完成artifacts

```text
experiments/EXP_F10/frozen_protocol.json
experiments/EXP_F10/support_dataset_manifest.jsonl
experiments/EXP_F10/support_samples/*.npz
experiments/EXP_F10/checkpoints/
experiments/EXP_F10/support_oof.csv
experiments/EXP_F10/confirmed_failures.jsonl
experiments/EXP_F10/decision_inputs/
experiments/EXP_F10/decision_manifest.jsonl
experiments/EXP_F10/selected_rollouts/
experiments/EXP_F10/selected_outcomes.jsonl
experiments/EXP_F10/oracle_rollouts/
experiments/EXP_F10/oracle_outcomes.jsonl
experiments/EXP_F10/metrics.json
experiments/EXP_F10/run_metadata.json
experiments/EXP_F10/audit.json
reports/EXP_F10_report.md
reports/next_exp_fromF10.md
```

F10只有在新visual support data、实际selected-only interventions和post-selection oracle全部完成并审计后才完成。现在立即执行。
