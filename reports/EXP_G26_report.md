# EXP_G26 Report: Uncertainty-Supported Transition Coordinates

## Conclusion

**NOT SUPPORTED.** The aligned uncertainty ensemble completed 8/10 ordered tasks, tying the direct uncertainty ensemble and losing to physical shooting and single raw pi0.5 at 9/10. The decisive mechanism ablation was even stronger: deliberately assigning each candidate another candidate's disagreement score completed 10/10. Correctly attributed latent uncertainty therefore cannot explain the best outcome. G26 demonstrates that bootstrap averaging can help over its 7/10 mean-only variant, but it does not establish a control-relevant latent coordinate.

## Models and protected selection

G26 reused the 1,242 audited G24 candidate branches in 414 matched decision groups. For each of ten outer leave-one-attempt-out folds it trained five-member aligned and direct-visual ensembles twice: once on eight attempts for nested calibration on the ninth training attempt, then again on all nine training attempts for the protected outer evaluation. All bootstrap samples were drawn at the decision-group level, preserving the three matched candidates together. This produced 100 calibration and 100 final checkpoints.

The ensemble risk combined normalized progress-score standard deviation, physical-outcome disagreement, and visual-transition disagreement. Each fold selected its risk coefficient from `[0, 0.25, 0.5, 1, 2]` without reading the outer held-out or prospective outcomes. Aligned folds selected values between 0.25 and 1.0; direct folds selected values between 0 and 2.0. On outer held-out groups, aligned uncertainty obtained 0.5710 ranking accuracy and 0.00103 regret, while direct uncertainty obtained 0.5912 accuracy and 0.01218 regret. The aligned ensemble's lower regret did not translate into superior control success.

## Prospective causal comparison

Ninety fresh rollouts used the ten canonical perturbed snapshots and autonomous state-window F3. Every learned controller requested three native pi0.5 proposals, computed its frozen objective, committed the selected bytes unchanged, observed the resulting physical/image state, and replanned. The shuffled ablation retained the exact ensemble predictions and risk magnitudes but cyclically attached risk to another candidate, directly testing whether correct candidate-specific uncertainty caused any gain.

| Method | Ordered success | Lift complete | Endpoint error | Jerk | Candidate steps |
|---|---:|---:|---:|---:|---:|
| aligned uncertainty coordinate | 8/10 | 10/10 | 0.08996 | 0.08608 | 0 |
| direct uncertainty ensemble | 8/10 | 10/10 | 0.09604 | 0.08817 | 0 |
| aligned ensemble mean only | 7/10 | 10/10 | 0.08058 | 0.10021 | 0 |
| shuffled latent disagreement | **10/10** | 10/10 | 0.05634 | 0.08406 | 0 |
| raw-history GRU | 8/10 | 10/10 | 0.07194 | 0.08834 | 0 |
| memoryless aligned visual | 8/10 | 10/10 | **0.05351** | **0.07741** | 0 |
| physical matched-branch shooting | 9/10 | 10/10 | 0.06288 | 0.08442 | 5,862 |
| single raw pi0.5 | 9/10 | 10/10 | 0.05715 | 0.08113 | 0 |
| initial-observation open loop | 0/10 | 0/10 | 1.06736 | 0.11452 | 0 |

The aligned risk penalty improves one task over aligned ensemble mean, but this is insufficient: it ties the direct ensemble, loses to simpler baselines, and is dominated by a semantically wrong risk assignment. The shuffled 10/10 result is real execution evidence but is not evidence for the uncertainty hypothesis. It most plausibly exposes candidate-set/order effects and high variance in independent pi0.5 proposal draws. G27 must test relative set structure and permutation equivariance before treating any index-coupled result as a mechanism.

## Audit and artifacts

The independent audit passed with zero failures. It loaded and recomputed all 200 ensemble checkpoints, checked 70,380 bootstrap group draws, recomputed 2,484 nested-calibration and 2,484 outer held-out samples, reconstructed 1,652 prospective ensemble objectives and 393 fresh raw-GRU updates, verified 3,281 native action commits and all ninety rollout chains, and rebuilt every aggregate.

- `experiments/EXP_G26/calibration_models/` and `fold_models/`
- `experiments/EXP_G26/model_selection.json` and `frozen_system_manifest.json`
- `experiments/EXP_G26/rollouts/`, `case_metrics.jsonl`, and `metrics.json`
- `experiments/EXP_G26/audit.json`
- `scripts/experiments/run_exp_g26_uncertainty_coordinates.py`
- `scripts/experiments/audit_exp_g26.py`

The post-EXP disk check left 849 GB free.

## Consequence for the system

G26 rules out candidate-specific bootstrap disagreement as the missing latent-control role. It also reveals that independent scalar scoring ignores the relative structure of each proposal set, making it vulnerable to candidate order and proposal-draw variation. G27 will replace independent scores with a permutation-equivariant set ranker trained on whole matched intervention groups and will compare aligned-coordinate and raw-action versions at equal capacity.
