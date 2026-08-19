# EXP_G21 Report: Adaptive Causal-F2 Trigger

## Conclusion

**NOT SUPPORTED.** The contrastive causal-regret latent completed 7/10 tasks, below the direct trigger and always-shooting at 8/10 and below single raw pi0.5 and a matched random trigger at 9/10. It saved branch execution relative to always-shooting, but the lower success means the preregistered efficiency condition is false. The simplest single-proposal closed-loop controller was both the most successful and the cheapest.

## New representation and control role

G21 reconstructed 1,859 pre-decision groups from audited G15--G18 matched branches. Each sample contained exact pre-decision EEF/book state, active action, the untouched first 10x7 pi0.5 proposal, realized scores/successes for all candidates, and candidate-zero regret. Three matched families were trained: a 16-D supervised contrastive regret latent, a 16-D autoencoding regret latent, and a direct 64-D MLP. Thresholds were frozen using held-out balanced accuracy with a branch-cost penalty.

The contrastive latent won the latent-family rule with balanced accuracy 0.6322 and regret MAE 0.03959. The direct MLP had higher balanced accuracy, 0.6373, but worse regret MAE, 0.04753. During prospective execution, adaptive methods queried three native proposals at each decision. When the trigger was false, no candidate branch was executed and proposal zero was committed unchanged. When true, all three prefixes were executed from the same complete checkpoint and state-value shooting selected the committed prefix.

## Results

| Method | Success | Candidate steps | Endpoint error down |
|---|---:|---:|---:|
| latent adaptive F2 | 7/10 | 3,024 | 0.063704 |
| direct adaptive F2 | 8/10 | 2,792 | 0.093750 |
| always three-proposal shooting | 8/10 | 6,411 | 0.069318 |
| single raw pi0.5 | **9/10** | **0** | **0.056011** |
| matched random trigger | **9/10** | 2,865 | 0.057850 |
| initial-observation open loop | 0/10 | 0 | 1.063902 |

The latent did reduce branch cost by 53% relative to always-shooting, but lost one task to always-shooting and two to single/random. Attempt 44 is especially informative: latent, direct, and always-shooting failed, whereas single and random-trigger succeeded. Thus causal candidate evaluation can steer the controller toward worse continuations even when its local scorer prefers them. The evidence favors simplifying F2 to single-proposal receding feedback for this cohort.

## Retry and audit

The first complete run deliberately logged random-trigger predicted regret as NaN to mean “not applicable,” which made the global finite invariant false. It is preserved but excluded. Retry1 regenerated the identical dataset/models, explicitly reused the 50 finite non-random rollout artifacts, and executed ten new random-trigger rollouts with a finite zero placeholder. The independent audit passed: it rebuilt all 1,859 samples, recomputed three models and 1,276 triggers, checked 2,107 proposal-to-commit chains and 60 rollouts, and verified the 50 reused plus ten new provenance records.

- `scripts/experiments/run_exp_g21_adaptive_f2_trigger.py`
- `scripts/experiments/audit_exp_g21.py`
- `experiments/EXP_G21_retry1/causal_regret_dataset.npz`
- `experiments/EXP_G21_retry1/regret_{contrastive,autoencoder,direct}.pt`
- `experiments/EXP_G21_retry1/model_selection.json`
- `experiments/EXP_G21/rollouts/` -- 50 finite formal source rollouts
- `experiments/EXP_G21_retry1/rollouts/` -- ten corrected random-trigger rollouts
- `experiments/EXP_G21_retry1/case_metrics.jsonl`, `metrics.json`, `audit.json`

The post-EXP disk check left 850 GB free; the two G21 directories occupy about 21 MB.

## Next decision

Repeated latent proposal scorers and gates have failed, and single-proposal receding feedback is now the strongest F2. G22 will freeze this simpler F2 and collect matched counterfactual switch-timing interventions. It will train an action-history latent directly on downstream regret of switching now versus continuing the active action, then test that latent as an autonomous temporal F3 against a direct-state regret model and the existing state-window F3.
