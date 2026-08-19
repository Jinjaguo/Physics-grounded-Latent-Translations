# EXP_G36 Report: Causal Physical-Effect Action Coordinate

## Conclusion

**NOT SUPPORTED.** The learned physical-effect coordinate detected all 60 imposed disturbances, but it did not improve recovery over the direct six-dimensional physical residual. It produced 23/30 strict `grasp -> lift -> place` successes versus 24/30 for the direct baseline, lost the paired comparison 1–2, incurred 1,895 rejected candidate steps versus 880, and had a 21.66% non-intervention false-trigger rate versus 8.53%. The direct physical F2 remains the justified recovery mechanism.

## New scientific content

G36 moved the latent away from the failed G35 phase/switching role and tested it at the causal F2 bottleneck. From the audited G24 matched-state branches it built 1,242 examples containing the current eight-dimensional physical state, a 35-dimensional native five-action prefix, and the realized six-dimensional end-effector/book displacement. Ten held-out attempts produced 30 independently trained checkpoints for three genuinely different representations:

1. an aligned six-dimensional action/effect coordinate with decoded displacement and contrastive pairing;
2. a predictive bottleneck with the same architecture but no contrastive loss;
3. a shuffled action/outcome coordinate that destroys causal pairing.

The fixed non-latent comparator was G33's direct displacement predictor. Every threshold was calibrated on the nine training attempts of its fold and frozen before prospective evaluation.

## Cross-fitted representation results

The aligned coordinate's mean held-out false-trigger rate was 14.99%, its decoded physical-displacement MAE was 0.00290, and its coordinate alignment RMSE was 0.1705. The predictive bottleneck obtained 12.84%, 0.00268, and 0.1817. Shuffling worsened displacement MAE to 0.01197 and alignment RMSE to 0.3653, confirming that the models used the true action/outcome pairing; however, this representational signal did not translate into better closed-loop control.

## Prospective causal evaluation

The frozen G35 state-based autonomous F3, three prompts, continuous current-state retargeting, ten untouched canonical starts, and common explicit policy noise were held constant. Eight methods ran for three repeats, yielding 240 new simulator rollouts. Every perturbed method received two real 3 cm book displacements. Every recovery decision executed a native five-step candidate, measured the realized next physical state, accepted it or restored the exact simulator/controller checkpoint, and continued from the accepted state.

| Method | Strict success | Disturbance recall | False-trigger rate | Rejected candidate steps |
|---|---:|---:|---:|---:|
| aligned effect latent | 23/30 | 60/60 | 21.66% | 1,895 |
| predictive bottleneck | 23/30 | 59/60 | 24.51% | 2,175 |
| shuffled-pair latent | 22/30 | 35/60 | 10.68% | 920 |
| **direct physical residual** | **24/30** | **60/60** | **8.53%** | **880** |
| physical oracle recovery | 24/30 | 60/60 | 0% | 300 |
| always rollback | 21/30 | 60/60 | 48.40% | 6,355 |
| no recovery | 4/30 | 0/60 | 0% | 0 |
| clean control | 21/30 | n/a | 0% | 0 |

The aligned latent's 23/30 result was three paired wins and three losses against the predictive bottleneck, four wins and three losses against the shuffled ablation, and one win versus two losses against both direct physical and oracle recovery. It therefore fails every part of the predeclared superiority rule except disturbance recall and superiority over no recovery.

## Mechanism finding

The learned coordinate contains causal pairing information but compresses it in a way that is poorly calibrated for branch acceptance. It rejects more than twice as many candidate steps as the direct residual while allowing no end-to-end gain. The comparison with no recovery still establishes the importance of causal F2: recovery raises strict success from 4/30 to 23–24/30. What fails is specifically the claim that a learned latent is the better verifier.

## Audit and artifacts

The independent audit passed with no failures. It recomputed all 30 fold models' saved predictions and thresholds, 3,726 held-out predictions, 97,425 proposal-coordinate predictions, 13,330 realized-effect trials and exact restores, 4,293 online state-F3 decisions, 32,475 explicit noise seeds, 420 physical interventions, 10,825 native commits, all 240 rollout chains, aggregate metrics, paired comparisons, and the negative conclusion.

- `experiments/EXP_G36/model_selection.json` and 30 fold checkpoints
- `experiments/EXP_G36/rollouts/` and `case_metrics.jsonl`
- `experiments/EXP_G36/metrics.json`, `run_metadata.json`, and `audit.json`
- `scripts/experiments/run_exp_g36_physical_effect_coordinate.py`
- `scripts/experiments/audit_exp_g36.py`
- `experiments/EXP_G36_smoke/` is preserved interface validation and is not counted as an EXP

The single post-EXP disk check reported 847 GB free.

## Consequence for the system

The final architecture should retain state F3 and direct physical-residual F2. G36 does not justify the physical-effect coordinate as a control module. To give the action-latent thesis a fair mechanism-level test without repeating the same task-5 residual formulation, G37 must evaluate whether a shared learned action representation provides transfer across genuinely different tasks, objects, and action semantics when task-specific direct predictors have little or no target-task data.
