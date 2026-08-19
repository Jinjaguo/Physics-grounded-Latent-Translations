# EXP_G14 Report: Outcome-Grounded Latent on Prospective Starts

## Conclusion

**SUPPORTED only by the preregistered switch-error tie-break; not yet supported as a success-rate contribution.** The newly trained outcome-grounded latent completed all 10/10 untouched official starts, as did state-only F3, old-latent F3, fixed-time F3, and the future-visible controller. Among the learned switchers, outcome-latent F3 had the smallest mean absolute switch error (8.5 steps versus 9.5 for old latent and 10.5 for state), so it wins the ordered G14 rule. It did not improve official success, and state-only F3 had the best endpoint error (0.031574 versus 0.033248). The cohort was too easy to establish the latent as a necessary system component.

## New representation and training

Fourteen task-5 training episodes produced 2,784 fixed 16-step action-history windows. A 16-D encoder was jointly trained to predict lift/place phase, five-step future end-effector and book displacement, and the next action. Its final training losses were phase BCE 0.008950, normalized motion MSE 0.053244, and next-action MSE 0.092601. A matched temporal outcome-latent F3 was then trained on 562 policy-issue windows and reached 1.0 training accuracy, precision, and recall. These are training-fit measurements, not held-out claims.

## Prospective execution

The exclusion manifest proves official init indices 40–49 were absent from all prior Wave19 train/development/test episodes. Ten new exact snapshots were captured after the recorded stabilization controls, and every method executed a fresh closed-loop LIBERO rollout from each snapshot.

| Method | Success | Switch error ↓ | Endpoint error ↓ | Candidate steps |
|---|---:|---:|---:|---:|
| outcome-latent F3 | 10/10 | **8.5** | 0.033248 | 3,705 |
| old-latent F3 | 10/10 | 9.5 | 0.034764 | 3,482 |
| state-only F3 | 10/10 | 10.5 | **0.031574** | 3,531 |
| fixed-time F3 | 10/10 | 11.5 | 0.038961 | 3,562 |
| future-visible full prompt | 10/10 | n/a | 0.038016 | 3,641 |

All methods lifted successfully, retained the object through composition, and produced finite traces. Outcome grounding improved switching timing under the declared tie-break, but the lack of any task failure means this result cannot satisfy the final requirement that the latent and integrated architecture make a strict, robust contribution.

## Audit and artifacts

The independent audit checked all 14 training memberships, the ten untouched indices and exact snapshots, 50 rollout files, proposal-selection-to-committed-action chains, and recomputed every aggregate. It passed with no failures. EXP_G14 occupies about 14 MB; the post-experiment disk check left 849 GB free.

- `scripts/experiments/run_exp_g14_outcome_latent.py`
- `scripts/experiments/audit_exp_g14.py`
- `experiments/EXP_G14/outcome_latent_dataset_manifest.json`
- `experiments/EXP_G14/outcome_latent.pt`, `temporal_f3_outcome_latent.pt`
- `experiments/EXP_G14/prospective_index_exclusion_manifest.json`
- `experiments/EXP_G14/snapshots/`, `rollouts/`, `case_metrics.jsonl`
- `experiments/EXP_G14/metrics.json`, `audit.json`, `run_metadata.json`

## Next decision

EXP_G15 will freeze these ten starts and introduce matched, physically valid XY object displacements before policy execution. It will compare the complete outcome-latent system with state-F3, F2-disabled single sampling, F3-disabled future-visible control, restart-after-switch, initial-observation open loop, and same-attempt unperturbed action replay. The new cohort must expose feedback and composition demands while preserving solvable initial physics.
