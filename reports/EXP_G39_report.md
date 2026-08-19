# EXP_G39 Report: Latent Corrective Proposal Generation

## Conclusion

**NOT SUPPORTED.** The conditional latent generator produced and executed new corrective controls, but it did not beat direct action regression. Across 30 matched perturbed three-stage rollouts, latent generation achieved 25/30 strict successes, direct generation 27/30, native five-proposal search 25/30, raw interpolation 25/30, pairwise native ranking 24/30, shuffled generation 22/30, and the declared physical oracle 26/30. The latent had one paired win and three losses against direct generation. It used fewer rejected candidate steps than native search (1,010 versus 1,820), but the success and paired criteria failed.

The strong novelty condition also failed. Independent reconstruction found a median nearest-native RMS distance of 0.06136 for the generated prefix, but a minimum of 0.00693 and only 58.57% of generated decisions above 0.05 RMS. The representation therefore sometimes generated a meaningful new control, but often collapsed close to an existing proposal. This is additional evidence against, not a reason to invalidate, the experiment.

## Hypothesis and new mechanism

G38 could only reorder five fixed native π0.5 proposals. G39 hypothesized that a conditional action latent trained on actual failed-then-recovered branches could decode a new five-action correction after candidate zero failed, exceed both a direct action regressor and native five-sample search, reduce rejected work, and lose its advantage when causal pairing was shuffled.

The experiment reconstructed 401 correction pairs from audited G36 branches. A pair was included only when candidate zero was rejected by the held-attempt direct physical detector and a later physically executed candidate was accepted. Every row retained current physical state, failed five-action prefix, accepted target prefix, attempt/repeat/decision provenance, and realized target score.

Ten held-attempt-out folds trained three families, for 30 saved checkpoints: a conditional VAE with an 8-D latent, a deterministic direct MLP, and the same VAE trained with shuffled correction targets. During prospective execution, candidate zero remained the exact native proposal. The learned or direct correction replaced candidate one, while native candidates one through three remained as fallbacks. Raw interpolation, G38 pairwise native ranking, unchanged five-sample native order, and physical-oracle recovery supplied mechanism baselines. All methods used G35 state F3, G33 direct physical verification, exact checkpoint restoration, common explicit π0.5 noise, and two real 3 cm book perturbations.

## Prospective results

| Method | Strict success | Official success | Premature grasp/lift | Rejected steps | Final error |
|---|---:|---:|---:|---:|---:|
| latent generator | 25/30 | 26/30 | 4 / 14 | **1,010** | 0.05375 |
| **direct generator** | **27/30** | **28/30** | 4 / 11 | 1,055 | **0.05345** |
| shuffled generator | 22/30 | 22/30 | 7 / 15 | 1,450 | 0.19439 |
| raw interpolation | 25/30 | 26/30 | 6 / 12 | 1,295 | 0.06871 |
| pairwise native | 24/30 | 25/30 | 2 / 12 | 1,620 | 0.05943 |
| five-sample native | 25/30 | 27/30 | 7 / 11 | 1,820 | 0.05633 |
| physical oracle | 26/30 | 26/30 | **0 / 0** | **300** | 0.06368 |

The shuffled generator's 22/30 result shows that causal correction pairing contains useful signal. The latent improved over shuffled by seven paired wins and four losses, but compression was not the useful part: direct generation achieved two more strict successes, slightly lower error, and a 3-to-1 paired advantage over latent. Latent tied native search in pooled strict success and had a 3-to-3 paired balance. Its rejected-work reduction is real, but cannot compensate for failure of the preregistered success rule.

## Mechanism finding

G39 removes the argument that earlier latent failures occurred only because the controller could not leave a fixed proposal set. The latent decoder did intervene in control space and those actions were actually executed, yet direct regression was better. The learned coordinate also partially collapsed toward native actions online. Together with G16, G17, G23, G30, G36, and G38, the result indicates a persistent mismatch: representation objectives can fit causal branch structure, but their bottleneck or decoder does not create a reliably better executable control distribution than direct state/action mechanisms and untouched π0.5 proposals.

## Audit and artifacts

The independent audit passed. It rebuilt all 401 correction pairs, reloaded 30 checkpoints, recomputed 1,203 held predictions, reconstructed 8,852 generated/reordered proposal sets and novelty distances, checked 44,260 acquisition noise seeds, 10,562 direct physical scores and exact restores, 420 interventions, 8,852 committed action chains, all 210 rollouts, aggregates, paired comparisons, and the negative conclusion. The completion disk check left 846 GB free.

- `experiments/EXP_G39/correction_dataset.npz`, `models/`, and `model_selection.json`
- `experiments/EXP_G39/rollouts/`, `case_metrics.jsonl`, `metrics.json`, and `audit.json`
- `scripts/experiments/run_exp_g39_latent_corrective_generation.py`
- `scripts/experiments/audit_exp_g39.py`
- `experiments/EXP_G39_smoke/` preserves pre-formal interface validation and is not a separate EXP

## Consequence for G40

No additional action-latent F2 family is justified before the requested stop at G40. The final experiment must freeze the best verified direct/state architecture and compare it in one matched causal protocol against the existing latent intervention plus open-loop/no-F2, teacher-forced progression, restart-from-initial, future-goal-unprotected, oracle-F3, no-F3, and no-checkpoint-memory mechanisms. Success of the direct system must be reported separately from the paper's stronger latent-coordinate claim.
