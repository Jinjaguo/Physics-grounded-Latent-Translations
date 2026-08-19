# EXP_G6 Report: Train-Only Closed-Loop Data Aggregation

## Conclusion

**NOT SUPPORTED: train-only corrective aggregation did not improve official success beyond the G4 2/10 baseline.** The selected correction-upweighted model reached 2/10, exactly tying G4, while worsening mean target error.

## Executed experiment

The frozen G4 state-plus-latent policy was rolled from two certified train checkpoints per LIBERO-10 task. This produced 20 real aggregation rollouts and 2,560 learner-visited states. Each state was paired with the same-step demonstrated train action; no development or test action entered fitting. Feature-distance diagnostics recorded how far the learner distribution moved from expert features.

Ten task-local models were refitted for each of two constructions: equal-weight expert/correction samples and threefold correction upweighting. Both were executed over the unchanged ten late development checkpoints and complete horizons.

| Method | Success | Selection tasks 0–4 | Confirmation tasks 5–9 | Mean final error ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|---:|
| G4 state+latent MLP | 2/10 | — | — | 0.443123 | 0.225582 |
| equal aggregation | 1/10 | 0 | 1 | **0.317599** | 0.108510 |
| correction ×3 | 2/10 | 1 | 1 | 0.527779 | **0.103512** |

Aggregation made actions smoother, and equal weighting reduced mean error, but neither construction exceeded the required success baseline. The correction label assumes that the source action at the same elapsed time is valid from a deviated state. The results show this is often false; real recovery requires a multimodal or short-horizon action-sequence model rather than additional weighting of phase-aligned labels.

## Audit and artifacts

The independent audit checked all 20 collection archives and 2,560 labels, loaded all 20 refitted checkpoints, reopened all 20 development rollouts, and exactly reproduced success/error aggregates. EXP_G6 occupies approximately 20 MB and left approximately 850 GB free.

- Runner/auditor: `scripts/experiments/run_exp_g6_data_aggregation.py`, `scripts/experiments/audit_exp_g6.py`
- Collection: `experiments/EXP_G6/collection_manifest.json`, `collection/*.npz`
- Training: `experiments/EXP_G6/checkpoints/*.pt`, `training_summary.json`
- Evaluation: `experiments/EXP_G6/rollouts/*.npz`, `case_metrics.jsonl`, `metrics.json`, `audit.json`

## Next decision

G7 will replace one-step regression with chunked multimodal action prediction. It will compare deterministic chunk regression and a mixture-density chunk model, execute only a short prefix before re-observing, and retain a matched non-latent ablation. This changes the control formulation rather than tuning G6 weights.
