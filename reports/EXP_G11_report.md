# EXP_G11 Report: Matched-Snapshot Post-Lift Place Isolation

## Conclusion

**SUPPORTED: an explicit completed-lift context repaired post-lift place control.** From five exact realized lifted states, `contextual_place_full_value` achieved 4/5 official successes while place-only and the original full prompt each achieved 3/5. All methods retained the lift in 5/5 cases.

**WEAKLY SUPPORTED BY THE DECLARED ERROR TIE-BREAKERS: post-lift suffix values and the action latent improved endpoint accuracy, but not official success.** Place-only full-value, suffix-state, and suffix-state+latent each achieved 3/5; mean final error improved from 0.05613 to 0.04817 to 0.04677. The latent contribution is therefore small and does not yet constitute a success-rate gain.

EXP_G11 establishes a viable 4/5 post-lift controller and removes the G10 place bottleneck. It does not yet execute autonomous switching end to end.

## New post-lift dataset and models

Fourteen task-5 Wave19 training episodes produced 1,646 post-lift suffix states. Each suffix began at the verified 4 cm×3-step book-height boundary, which occurred at training action indices 71–92, and ended at official success. Two task-local values used identical state features and normalized suffix-progress targets; the matched latent value additionally received the frozen 32-D action latent.

The state model used 385 inputs and reached final train MSE 0.000647. The state+latent model used 417 inputs and reached 0.000340. No development or test episode entered fitting.

## Matched causal construction

For each of the five development starts, G11 executed the protected lift once with the G9 two-proposal causal state-value controller. The oracle boundary was reached in 70–84 committed actions. The complete simulator/controller snapshot, current observation payload, lift trace, committed action history, and latent were saved at that boundary.

All five place methods then restored the exact same snapshot and history for that episode. This removes the lift-state variation that confounded G10, where each method independently sampled a different lift. Every place decision again executed both π0.5 candidates from a recoverable snapshot, scored realized feedback, restored, and committed the winner.

| Place branch | Official success | Lift retained | Final error ↓ | Jerk ↓ | Candidate steps |
|---|---:|---:|---:|---:|---:|
| place-only + full value | 3/5 | 5/5 | 0.056133 | 0.041796 | 927 |
| **completed-lift contextual prompt + full value** | **4/5** | 5/5 | 0.047209 | 0.044525 | 905 |
| original full prompt + full value | 3/5 | 5/5 | 0.054056 | **0.040516** | 926 |
| place-only + suffix state value | 3/5 | 5/5 | 0.048170 | 0.043506 | 921 |
| place-only + suffix state+latent value | 3/5 | 5/5 | **0.046773** | 0.041800 | 908 |

The matched construction itself is a major finding: post-lift place became 3–4/5 rather than G10’s 0/5 hard-gated result. Contextual language was the only mechanism to improve official success. Suffix values changed proposal ranking and endpoint error, but neither state nor latent suffix ranking increased the number of completed tasks.

## Audit and artifacts

The independent audit loaded both suffix checkpoints, verified all 14 fitting episodes were training-only, reopened five exact switch snapshots, reconstructed five protected-lift action chains and histories, proved all five methods shared identical initial physical states within each episode, and reconstructed 25 place action chains and metrics. It passed without failures.

EXP_G11 occupies approximately 6.0 MB and left 849 GB free.

- `scripts/experiments/run_exp_g11_post_lift_place.py`
- `scripts/experiments/audit_exp_g11.py`
- `experiments/EXP_G11/suffix_dataset_manifest.json`, `training_summary.json`
- `experiments/EXP_G11/suffix_progress_value_state.pt`, `suffix_progress_value_state_latent.pt`
- `experiments/EXP_G11/switches/`, `matched_switch_branch_manifest.json`
- `experiments/EXP_G11/rollouts/`, `case_metrics.jsonl`
- `experiments/EXP_G11/metrics.json`, `audit.json`, `run_metadata.json`

## Next decision

The remaining core bottleneck is autonomous F3. EXP_G12 will train temporal completion models from train-only policy-issue sequences and execute full protected `lift -> contextual place` rollouts with learned state-window and state+latent-window switching, compared against oracle, train-median fixed-time, and unrestricted future-visible mechanisms.
