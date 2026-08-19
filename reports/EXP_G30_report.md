# EXP_G30 Report: Generative Latent Action Refinement

## Conclusion

**NOT SUPPORTED.** Latent-coordinate CEM generated and executed genuinely new native action prefixes, but failed all 30 ordered tasks. Equal-budget raw-action CEM completed 6/30, latent random search completed 1/30, and merely encoding/decoding the base action completed 19/30. The unmodified single pi0.5 proposal completed 26/30. The action coordinate is therefore not a safe generative control space under this learned short-horizon utility model.

## Models and search protocol

G30 trained 60 models: three-member latent and raw ensembles in ten leave-one-attempt-out folds. The latent model encoded a normalized five-step 35-D action prefix into 8-D, decoded it back to native action space, predicted the realized eight-dimensional physical/utility/success effect, and scored state-coordinate utility. The raw model predicted the same effects and utility directly from state plus the 35-D prefix.

Held-out ranking accuracy was 0.7371 for latent versus 0.7136 for raw; latent mean regret was 0.00572 versus 0.01227. Mean normalized reconstruction error across folds was roughly 0.10-0.16. Cross-fitted selection chose a large four-times median within-group radius for both search spaces. Latent used support penalty 0.05, no pessimism and zero gate threshold; raw used no support penalty, beta 1.0 and threshold 0.2. These settings were frozen before testing.

At every decision, latent CEM evaluated 32 coordinate samples for three iterations, decoded the best into an exact five-step action prefix, and compared it with candidate zero. Latent random used the same total 96 model evaluations in one pass. Raw CEM used 32x3 evaluations in normalized 35-D action space. No search candidate was executed in the simulator until the selected decoded/refined prefix was committed.

## Prospective results

G30 executed 240 fresh simulator rollouts across three method-neutral common-noise repeats.

| Method | Ordered success | Per-repeat | Lift complete | Endpoint error | Jerk | Generated selections |
|---|---:|---:|---:|---:|---:|---:|
| latent-coordinate CEM | 0/30 | 0, 0, 0 | 8/30 | 0.71104 | 0.27503 | 1,799/1,800 |
| latent random search | 1/30 | 0, 0, 1 | 8/30 | 0.69839 | 0.27581 | 1,751/1,758 |
| raw-action CEM | 6/30 | 1, 1, 4 | 23/30 | 0.39688 | 0.59846 | 1,617/1,618 |
| latent reconstruction | 19/30 | 7, 5, 7 | 28/30 | 0.14838 | 0.09653 | 1,421/1,421 |
| G28 aligned native point | 25/30 | 9, 8, 8 | 30/30 | **0.06241** | 0.08435 | n/a |
| physical shooting | 25/30 | 9, 8, 8 | 30/30 | 0.07254 | 0.08747 | n/a |
| single raw pi0.5 | **26/30** | 10, 8, 8 | 30/30 | 0.09391 | **0.08452** | n/a |
| initial-observation open loop | 0/30 | 0, 0, 0 | 0/30 | 1.07193 | 0.11341 | n/a |

Latent CEM had zero paired wins against every method. It lost six cases to raw CEM, 19 to reconstruction, 25 to aligned point and physical shooting, and 26 to single pi0.5. Its mean jerk more than tripled relative to the native policy. The gate selected generated actions on essentially every decision, so this is not a hidden fallback result. Iterative optimization systematically exploited value/effect model error outside the narrow causal branch support; the raw search was also unsafe but materially less catastrophic.

## Audit and artifacts

The independent audit passed with zero failures. It checked all 60 model members and their training logs, recomputed 828 held-out group-family decisions and both full search sweeps, regenerated 6,597 seeded CEM/random decisions and 6,588 selected decoded prefixes, regenerated 26,689 policy-noise seeds, verified 10,316 native commits and all 240 rollout chains, and rebuilt every aggregate and paired comparison.

- `experiments/EXP_G30/fold_models/`, `model_selection.json`, and `common_noise_manifest.json`
- `experiments/EXP_G30/rollouts/`, `case_metrics.jsonl`, and `metrics.json`
- `experiments/EXP_G30/audit.json`
- `scripts/experiments/run_exp_g30_generative_latent_refinement.py`
- `scripts/experiments/audit_exp_g30.py`

The post-EXP disk check left 849 GB free.

## Consequence for the system

The project should not treat this learned action autoencoder as an open-loop optimization manifold. G31 moves the representation after intervention rather than before it: use actual short-horizon execution, re-encoded realized visual/action effects, and complete simulator/controller checkpoints to accept a candidate or roll back and try another. This tests latent feedback as a recovery signal while keeping executed actions on the pi0.5 policy support.
