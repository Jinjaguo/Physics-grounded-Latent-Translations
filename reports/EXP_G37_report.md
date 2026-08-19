# EXP_G37 Report: Cross-Task Low-Data Action-Latent Transfer

## Conclusion

**NOT SUPPORTED.** The shared language-conditioned action/effect latent did not provide a cross-task control advantage. At the frozen 12-row target adaptation budget it achieved 6/12 held-out task successes, exactly tying pooled direct regression, the no-language latent, and the shuffled-pair latent. It used 1,995 rejected candidate steps versus 1,945 for pooled direct and 1,125 for the shuffled latent. Its disturbance recall was 17/24 with a 72.80% online false-trigger rate. The intended language-conditioned transfer contribution is absent.

## New tasks and causal data

G37 changed both task distribution and intervention target. It selected three distinct LIBERO-10 tasks with complete train/development/test membership and exact branch checkpoints:

1. task 0: put alphabet soup and tomato sauce in a basket;
2. task 3: put a black bowl in the bottom drawer and close it;
3. task 8: put two moka pots on a stove and turn it on.

From six train-only branch checkpoints per task, the collector automatically chose the most reference-moving free-joint object, generated five explicit-noise batches of three native π0.5 proposals, restored the exact checkpoint before every candidate, and executed five controls. Four batches were clean and the fifth moved the chosen object exactly 3 cm before execution. This produced 270 new matched-state causal rows: 216 clean and 54 physically disturbed. Each row stores task semantics, state, 35-D native action prefix, realized end-effector/object displacement, target body/joint, intervention, execution, and source membership.

## Models and adaptation experiment

Three leave-one-task-out folds compared target adaptation budgets 0, 12, and 36 within this EXP. Forty-two checkpoints covered:

- shared language-conditioned action/effect latent;
- pooled direct physical regression with task semantics;
- target-task direct regression when target rows existed;
- shared latent with language removed;
- shuffled action/outcome latent.

At zero target rows, shared latent disturbance recall was 55.56%, pooled direct 74.07%, no-language latent 92.59%, and shuffled latent 29.63%, all at a calibrated 5.56% clean false-trigger rate. At 12 rows, shared, pooled direct, and no-language each obtained 74.07% offline recall. At 36 rows, shared rose to 88.89%, but pooled direct had lower clean displacement MAE (0.00483 versus 0.00624) and the no-language latent had higher recall (96.30%). Language conditioning never supplied the decisive transfer signal.

## Prospective causal evaluation

The 12-row operating point was frozen before test. Two untouched test starts per task, two explicit-noise repeats, and seven mechanisms produced 84 causal rollouts. Every rollout received two real 3 cm target-object interventions. Rejected candidates were removed with exact simulator/controller restore and control continued from the accepted candidate's physical state.

| Method | Success | task 0 / 3 / 8 | Recall | Online false triggers | Rejected steps |
|---|---:|---:|---:|---:|---:|
| shared latent | 6/12 | 2 / 4 / 0 | 17/24 | 72.80% | 1,995 |
| pooled direct | 6/12 | 2 / 4 / 0 | 17/24 | 72.09% | 1,945 |
| task direct | 5/12 | 2 / 3 / 0 | 19/24 | 82.80% | 2,590 |
| no-language latent | 6/12 | 2 / 4 / 0 | 17/24 | 72.60% | 1,935 |
| shuffled latent | 6/12 | 2 / 4 / 0 | 15/24 | **51.56%** | **1,125** |
| physical oracle | 5/12 | 2 / 3 / 0 | 24/24 | 0% | 120 |
| no recovery | 4/12 | 3 / 1 / 0 | 0/24 | 0% | 0 |

Shared latent tied pooled direct, no-language, and shuffled on all 12 paired cases. It exceeded target-specific direct by one paired win and no losses, but failed the required superiority over both direct baselines and failed both language/pairing ablations. Task 8 was an execution-floor limitation: even oracle recovery completed 0/4, so no representation claim is made from that task's downstream failures.

## Evidence repair and audit

The first formal archives omitted terminal simulator state and per-candidate success/done. `EXP_G37_repair1` added them, revealing that `capture_snapshot()` performs a MuJoCo forward pass that can alter contact-dependent success at boundary states. `EXP_G37_repair2` preserved the raw pre-forward integration state, but LIBERO's `In` and `On` predicates depend on derived contacts not contained in `mjSTATE_INTEGRATION`. `EXP_G37_repair3` therefore also saved every official atomic goal predicate immediately after each `env.step()`. All directories are preserved; none is counted as a separate EXP.

The final independent audit passed. It rebuilt all 270 rows, checked 54 collection interventions, recomputed all 42 models and 1,890 calibration predictions, recomputed 43,080 proposal/effect predictions, verified 8,616 explicit noise seeds, 4,814 candidate executions, exact restores and official goal-predicate conjunctions, 168 prospective interventions, 2,872 native commits, all 84 rollout chains, aggregate metrics, and the negative conclusion.

- `experiments/EXP_G37/`: causal dataset, 42 models, original prospective evidence
- `experiments/EXP_G37_repair3/`: final auditable prospective evidence and `audit.json`
- `experiments/EXP_G37_repair1/`, `EXP_G37_repair2/`: preserved evidence-interface failures
- `scripts/experiments/run_exp_g37_cross_task_transfer.py`
- `scripts/experiments/run_exp_g37_prospective_repair.py`
- `scripts/experiments/audit_exp_g37.py`

The single post-EXP disk check reported 847 GB free.

## Consequence for the system

The direct physical F2 and state F3 remain the justified integrated mechanisms. Learned action latents have now failed as phase coordinates, physical residual verifiers, and cross-task transfer representations. G38 must stop asking the latent to duplicate a calibrated physical residual. It will instead test whether a causal action-support latent can change which native proposal is executed after a detected failure, with the direct residual retained as verifier.
