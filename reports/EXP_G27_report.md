# EXP_G27 Report: Permutation-Equivariant Proposal-Set Coordinates

## Conclusion

**NOT SUPPORTED.** The aligned-coordinate set ranker completed 8/10 ordered tasks. The equal-parameter raw-action set ranker and the aligned point-ranker ablation each completed 9/10. Although the aligned set model had the best held-out ranking accuracy, its set context reduced rather than improved prospective success. Permutation equivariance was implemented and verified, but it is not the missing latent-control mechanism.

## Set-wise models

G27 rebuilt whole three-candidate groups from all 1,242 G24 physical branches and 414 decisions. Ten leave-one-attempt-out folds trained three rankers: an aligned-coordinate DeepSets model, an architecturally identical raw-action DeepSets model, and an aligned candidate-independent point MLP. The two set models had equal parameter counts in every fold. All models used group-level progress regression plus best-candidate cross-entropy.

Offline, aligned set ranking reached 0.7685 accuracy and 0.01150 regret, versus 0.7285/0.01173 for raw set and 0.7495/0.01221 for the aligned point model. All six permutations of every held-out proposal set were evaluated. Maximum restored-score error was (4.77\times10^{-7}) for aligned set, (7.15\times10^{-7}) for raw set, and (4.77\times10^{-7}) for the point model.

## Prospective results

One hundred fresh autonomous lift-to-place rollouts were executed from the ten canonical snapshots. At every set-ranked decision the controller scored all six candidate permutations, restored scores to canonical order, saved the comparison, selected one exact native proposal and replanned from realized feedback.

| Method | Ordered success | Lift complete | Endpoint error | Jerk | Candidate steps |
|---|---:|---:|---:|---:|---:|
| aligned-coordinate set ranker | 8/10 | 10/10 | 0.06404 | 0.08736 | 0 |
| raw-action set ranker | **9/10** | 10/10 | 0.05735 | 0.07821 | 0 |
| aligned point ranker | **9/10** | 10/10 | 0.06331 | 0.08353 | 0 |
| aligned uncertainty ensemble | 7/10 | 10/10 | 0.06519 | 0.08099 | 0 |
| shuffled latent disagreement | **9/10** | 10/10 | 0.05526 | 0.08489 | 0 |
| raw-history GRU | 7/10 | 10/10 | 0.07367 | **0.07775** | 0 |
| memoryless aligned visual | 8/10 | 10/10 | **0.05444** | 0.07694 | 0 |
| physical matched-branch shooting | **9/10** | 10/10 | 0.06659 | 0.08049 | 6,332 |
| single raw pi0.5 | 7/10 | 10/10 | 0.06609 | 0.08131 | 0 |
| initial-observation open loop | 0/10 | 1/10 | 1.04620 | 0.11367 | 0 |

The live maximum permutation error was only (5.66\times10^{-7}), so the negative result is not an order bug in DeepSets. Instead, the candidate-independent aligned point model beats aligned set context, and raw set beats aligned set. The strongest justified system remains a simple native-action ranker or raw representation; no action-coordinate contribution is established.

## Audit and artifacts

The independent audit passed with zero failures. It checked thirty fold checkpoints, recomputed 1,242 held-out decisions and 7,452 held-out permutations, rebuilt 1,238 live set decisions and 7,428 live permutations from base predictor features, recomputed 1,247 ensemble/GRU baseline decisions, verified 3,778 native commits and all one hundred rollout chains, and reconstructed all ten aggregates.

- `experiments/EXP_G27/fold_models/` and `model_selection.json`
- `experiments/EXP_G27/rollouts/`, `case_metrics.jsonl`, and `metrics.json`
- `experiments/EXP_G27/audit.json`
- `scripts/experiments/run_exp_g27_permutation_set_ranker.py`
- `scripts/experiments/audit_exp_g27.py`

The post-EXP disk check left 849 GB free.

## Consequence for the system

G27 separates two issues. Set rankers are genuinely permutation equivariant, so G26's shuffled result was not evidence that index-aware control is desirable. Yet controller success continues to vary materially across fresh experiments because methods consume independent diffusion-noise proposal draws. G28 will add explicit noise-conditioned batch inference and use common random numbers across aligned, unaligned, and raw equal-capacity point rankers over three full repeats. This will estimate representation effects without conflating them with proposal sampling luck.
