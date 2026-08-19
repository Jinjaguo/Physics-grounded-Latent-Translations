# EXP_G4 Report: State-Conditioned Residual Imitation

## Conclusion

**SUPPORTED: observation/state conditioning produced the first non-oracle long-horizon atomic completions, and the matched state-plus-latent MLP outperformed the state-only MLP.** The result is still insufficient for the final system: success was only 2/10, action jerk increased sharply, and no autonomous switching or composition was tested.

## Hypothesis and new mechanism

EXP_G3 showed that the action-only latent and four-step oracle path score could not identify long-horizon contact phase. EXP_G4 therefore replaced candidate shooting with a new learned controller family. Ten task-local policies were fitted from Wave19 train episodes using the realized robot state, end-effector pose, gripper state, object-body pose/velocity, previous action, and normalized episode progress. Three state-conditioned families were compared:

- nearest-state action retrieval;
- a state-only MLP;
- the same MLP augmented by the frozen rolling 16-step action latent.

Each learned policy recomputed its action after every actual LIBERO step. After the branch point, no state, action, phase observation, or latent was teacher-forced from the source trajectory. Source actions remained an oracle reference only.

## Data, fitting, and evaluation

- Training: all 140 frozen Wave19 train episodes, 14 per LIBERO-10 task.
- Development evaluation: the exact ten late checkpoints and complete remaining horizons used by EXP_G3.
- Test split: unopened.
- State/action alignment: every saved trajectory had `N` controls and `N+1` physical states; state `t` was paired with action `t`.
- Models: 20 independently fitted MLP checkpoints, one state-only and one state-plus-latent model per task; 30 epochs, batch size 256, AdamW learning rate 0.001.
- Baselines: source oracle, frozen receding F1, and state retrieval.

The initial CUDA attempt failed before training because the restricted process could not initialize the NVIDIA driver despite PyTorch capability reporting. That empty attempt is preserved under `experiments/debug/EXP_G4_attempt1_cuda_driver`. The complete experiment ran on CPU without changing the model or protocol.

## Results

| Method | Official success | Mean final target error ↓ | Mean action jerk ↓ |
|---|---:|---:|---:|
| source oracle | 5/10 | 0.000785 | 0.116032 |
| receding F1 | 0/10 | 0.293316 | 0.083209 |
| state retrieval | 0/10 | **0.276952** | 0.083973 |
| state-only MLP | 1/10 | 0.468286 | 0.231406 |
| **state-plus-latent MLP** | **2/10** | 0.443123 | 0.225582 |

State conditioning raised official completion from the EXP_G3 deployable result of 0/10 to nonzero success. Under a capacity- and training-matched comparison, adding the frozen action latent raised success from 1/10 to 2/10 and slightly reduced both mean error and jerk. This is direct evidence that the latent contributed control-relevant information in this controller, not merely an offline retrieval score.

The error metrics also reveal a tradeoff. Retrieval produced no successes but the lowest average failure endpoint error and smooth actions. Both MLPs occasionally completed contact-rich tasks but produced substantially higher jerk and worse mean error. A successful next mechanism must retain sequence/contact information without making stepwise corrections unstable.

## Integrity and artifacts

`scripts/experiments/audit_exp_g4.py` loaded all 20 model checkpoints and reopened all 50 committed rollouts. It independently recomputed official success and final physical error from the original branch reference states and matched every saved aggregate with zero discrepancies.

- Runner/auditor: `scripts/experiments/run_exp_g4_state_policy.py`, `scripts/experiments/audit_exp_g4.py`
- Training records/checkpoints: `experiments/EXP_G4/training_summary.json`, `experiments/EXP_G4/checkpoints/*.pt`
- Exact train membership and run metadata: `experiments/EXP_G4/run_metadata.json`, `environment.json`
- Per-case and step-level evidence: `experiments/EXP_G4/case_metrics.jsonl`, `experiments/EXP_G4/rollouts/*.npz`
- Aggregate/audit: `experiments/EXP_G4/metrics.json`, `audit.json`

EXP_G4 occupies approximately 43 MB. The completion-time disk check showed approximately 850 GB free.

## Claim boundary and next decision

EXP_G4 establishes neither robust atomic control nor the final two-action system. Eight of ten state-plus-latent cases failed, and the controller lacks persistent learned temporal memory. The next scientific bottleneck is error accumulation and phase aliasing across a long manipulation sequence.

EXP_G5 will train recurrent state-only and state-plus-latent sequence policies. Their hidden state will be initialized from the real pre-branch episode history; after branching, each update will consume only the realized current state and executed-action latent. This directly tests whether persistent temporal context improves completion and whether the latent contribution survives a stronger matched sequence baseline.
