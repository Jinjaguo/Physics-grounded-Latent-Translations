# EXP_G33 Report: Action-Coordinate Disturbance Memory

## Conclusion

**NOT SUPPORTED: the aligned action-latent residual is not the best disturbance memory. SUPPORTED: action-conditioned physical residuals enable efficient checkpoint recovery under real perturbations.** The latent detector recalled only 30/60 injected disturbances and completed 14/30 ordered tasks. Raw-action visual residuals recalled 39/60 and completed 15/30. The conventional raw-action physical residual detected all 60/60 disturbances and completed 24/30, with ten paired wins and no losses against latent recovery. The latent-memory hypothesis is decisively falsified.

## New intervention and calibration

G33 extended the causal rollout interface with a pre-candidate intervention hook placed after exact checkpoint restoration and before candidate execution. At decisions 8 and 24, candidate zero received an actual 3 cm XY displacement of the black-book free joint, with balanced cardinal direction and opposite later displacement. MuJoCo kinematics and observations were regenerated, and every event stored pre/post integration state, object pose, dual-view pixels, active stage, requested offset, candidate, and decision. A rejected disturbed trial was removed by exact restore before executing the next native proposal.

Ten leave-one-attempt-out G24 folds calibrated 95th-percentile clean residual thresholds for three models. The aligned model predicted dual-view latent change through a 16-D learned action coordinate. The raw visual model predicted the same latent change directly from the 35-D native action prefix. The physical model predicted six-dimensional end-effector/book displacement directly from state and native action. Calibration scores and thresholds were frozen before perturbed testing.

## Prospective causal protocol

Three explicit-noise repeats across ten canonical starts compared eight controllers, for 240 new rollouts. All used autonomous state F3, native π0.5 proposal order, three candidates and five-step replanning. Detector methods executed a proposal after a complete checkpoint, observed the actual disturbed or clean result, then accepted the reached state or restored and retried. Stage utility reused G32's frozen thresholds. Oracle rejected exactly disturbed trials; always rollback rejected candidate zero at every decision; no recovery accepted every first trial; the clean control used identical adaptive machinery without perturbations.

| Method | Disturbance recall | False-trigger rate | Ordered success | Per-repeat | Rejected steps |
|---|---:|---:|---:|---:|---:|
| aligned action-latent residual | 30/60 (0.50) | 0.1454 | 14/30 | 5, 5, 4 | 1,325 |
| raw-action visual residual | 39/60 (0.65) | 0.1528 | 15/30 | 4, 7, 4 | 1,410 |
| **raw-action physical residual** | **60/60 (1.00)** | **0.1451** | **24/30** | **8, 8, 8** | 1,310 |
| stage direct utility | 46/60 (0.77) | 0.9183 | 19/30 | 6, 7, 6 | 11,920 |
| oracle disturbance trigger | 60/60 (1.00) | 0 | 23/30 | 8, 8, 7 | **300** |
| always rollback candidate zero | 60/60 (1.00) | 0.4838 | 23/30 | 8, 8, 7 | 6,405 |
| no recovery | 0/60 | 0 | 4/30 | 1, 2, 1 | 0 |
| clean, no intervention | n/a | 0 | 23/30 | 8, 8, 7 | 0 |

Physical residual recovery gained 20 successes over no recovery and one over oracle/always. The one-success difference from oracle is not evidence that imperfect detection is intrinsically superior: after rollback, a different stochastic candidate can succeed or fail. It does show that the learned physical residual attained the strongest observed task result at much lower cost than always rollback. All detected perturbations incurred five action steps of recovery latency.

## Mechanism failure analysis

The aligned action-coordinate visual prediction did not isolate the controlled object displacement. Its 0.50 recall was below raw visual's 0.65 and physical's 1.0, while its false-trigger rate remained 0.145. Visual residual false triggers changed proposal order repeatedly, increased jerk to 0.1157, and prolonged trajectories to 7,553 committed steps. Physical residual retained direct action conditioning but measured the task-relevant eef/book consequence, yielding both perfect event recall and stable 8/10 success in every repeat.

Stage utility was also unsuitable as a general anomaly detector: it triggered 2,384 rollbacks, only 46 of which corresponded to the 60 disturbances, for 1.9% precision and 11,920 rejected steps. G32's clean-branch stopping advantage therefore does not transfer to selective perturbation detection.

The intervention itself is scientifically consequential. No recovery collapsed to 4/30 versus 23/30 for the matched clean control, while oracle rollback restored 23/30. This supplies direct `intervention -> executed disturbed action -> observed feedback -> restore/retry -> final metric` evidence for checkpoint recovery. It is rollback-based branch recovery, not a claim of physically executing a return trajectory.

## Audit and artifacts

The independent audit passed. It rebuilt all ten detector calibration folds, recomputed 29,889 action-coordinate predictions and 12,113 detector trials from saved pixels/models, recomputed 3,789 direct-utility trials, checked 420 physical perturbations, 15,902 exact restore events, 34,104 explicit-noise seeds, 11,368 committed native prefixes, all 240 rollout chains, and every detector/task/paired aggregate.

- `experiments/EXP_G33/calibration.json`, `calibration_scores.npz`, and `intervention_manifest.json`
- `experiments/EXP_G33/rollouts/`, `case_metrics.jsonl`, `metrics.json`, and `audit.json`
- `scripts/experiments/run_exp_g33_action_coordinate_disturbance_memory.py`
- `scripts/experiments/audit_exp_g33.py`
- the optional pre-candidate hook in `scripts/experiments/run_exp_g14_outcome_latent.py`
- `experiments/EXP_G33_smoke/` preserves the one-start interface validation and is not counted as an EXP

The post-EXP disk check left 848 GB free.

## Consequence for the system

The best perturbation recovery mechanism is now raw-action physical residual triggering plus exact checkpoint rollback. Aligned action coordinates should be removed from disturbance detection. Because action latents have now failed proposal ranking, generative search, trial verification, and perturbation detection, G34 must not tune another F2 residual. It will test a genuinely different learned representation: a monotonic, language-conditioned execution-phase coordinate trained with temporal ordering and evaluated as autonomous F3 under the fixed physical-recovery backbone.
