# EXP_G7 Report: Multimodal Action-Chunk Control

## Conclusion

**NOT SUPPORTED: neither chunk regression nor mixture-density prediction exceeded the G4 2/10 completion baseline.** State-only chunk control tied 2/10, latent chunk control reached 1/10, and the four-mode latent mixture reached 0/10.

## Experiment and results

Thirty task-local models were trained on Wave19 train trajectories: state-only deterministic 16-step chunks, state-plus-latent deterministic chunks, and state-plus-latent four-mode diagonal-Gaussian mixtures. At deployment each model executed four controls, observed the realized LIBERO state, updated the latent, and replanned without oracle future input.

| Method | Success | Mean final error ↓ | Raw action jerk ↓ |
|---|---:|---:|---:|
| G4 state+latent one-step MLP | 2/10 | 0.443123 | 0.225582 |
| state chunk | 2/10 | 0.343144 | 14.816056 |
| latent chunk | 1/10 | **0.257529** | 0.235738 |
| latent mixture | 0/10 | 0.261955 | **0.218138** |

Chunking improved failure endpoint error but not task completion. The mixture likelihood did not yield useful online mode choices. Latent contribution and multimodality hypotheses both failed.

The state-chunk raw jerk is dominated by unbounded gripper-logit magnitude even though LIBERO uses only its sign. This does not invalidate its two physical successes, but it disqualifies the output as a clean controller. Future controllers must emit the actual gripper sign command before continuity metrics are computed.

## Audit and artifacts

The independent audit loaded 30 checkpoints, reopened 30 long rollouts, and exactly reconstructed success and endpoint aggregates. EXP_G7 occupies approximately 36 MB; approximately 849 GB remained free.

- `scripts/experiments/run_exp_g7_chunk_policy.py`
- `scripts/experiments/audit_exp_g7.py`
- `experiments/EXP_G7/checkpoints/`, `rollouts/`, `training_summary.json`
- `experiments/EXP_G7/case_metrics.jsonl`, `metrics.json`, `audit.json`

## Next decision

The state/action-only families have plateaued at 2/10. G8 will use the existing real dual-camera policy-issue data: processed agent/wrist RGB, 8-D robot state, and 10-step controls. A shared visual chunk policy will be compared with a matched visual-plus-action-latent policy and executed at the native five-step replan interval.
