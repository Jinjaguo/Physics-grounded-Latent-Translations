# EXP_G3 Report: Long-Horizon Atomic Completion under Oracle F3

## Conclusion

**NOT SUPPORTED: the EXP_G2 causal-latent controller did not convert its 16-step local advantage into official long-horizon action completion.** All deployable learned/state controllers completed 0/10 development tasks. The recorded source controller completed 5/10 within the exact evaluated remaining horizons.

## Scientific question and protocol

EXP_G3 changed the evaluation protocol from a fixed 16-step local target to the complete remaining manipulation phase. It selected one development episode per LIBERO-10 task and used the latest certified branch with at least 128 future steps. Horizons were case-specific and ranged from 128 to 236 controller steps. Oracle F3 remained in force: the official success predicate terminated a rollout, and the saved source physical trajectory supplied dense current-action targets. Future source actions remained excluded from deployable candidate sets.

Five methods were executed from every matched full-state checkpoint: source oracle, receding F1, receding F2, causal state shooting, and causal latent shooting. Both causal controllers executed every four-step proposal prefix through the real simulator, restored the current checkpoint, committed the minimum-score candidate, observed the realized state, re-encoded the executed-action history, checkpointed, and replanned.

## Actual execution scale

- 50 committed long-horizon rollouts.
- 8,198 committed simulator/controller steps.
- 4,510 matched-state candidate interventions.
- 18,040 candidate simulator/controller steps.
- 820 causal selection decisions.
- All saved numerical values finite; no controller process failure.

## Results

| Method | Success | Mean final target error ↓ | Mean progress ↑ | Mean jerk ↓ |
|---|---:|---:|---:|---:|
| source oracle | 5/10 | 0.000785 | 0.371975 | 0.116032 |
| **receding F1** | 0/10 | **0.293316** | -0.206157 | 0.083209 |
| receding F2 | 0/10 | 0.403978 | -0.117704 | 0.084470 |
| causal state | 0/10 | 0.303720 | **0.065048** | **0.017918** |
| causal latent | 0/10 | 0.341637 | -0.045321 | 0.041326 |

All deployable methods tied at zero official success, so the primary hypothesis failed. Receding F1 had the lowest failure endpoint error, while causal state was the only deployable method with positive mean physical progress and had the smoothest actions. Causal latent was worse than both on their respective strengths.

The causal-latent selector chose direct controls 269/410 times, F1 32 times, F2 40 times, latent copy 25 times, and other direct variants 44 times. Its strong preference for locally safe direct continuation accumulated long-horizon drift. Four-step oracle path tracking was therefore myopic: it optimized local proximity without learning the contact/manipulation sequence needed for completion.

The source oracle succeeded on tasks 0, 2, 5, 6, and 8. Its 5/10 rather than 10/10 rate is consistent with truncating horizons to a multiple of the four-step planning prefix and with success timing at the very end of some continuations; it nevertheless establishes a large causal gap between viable recorded control and every deployable method.

## Integrity audit

`scripts/experiments/audit_exp_g3.py` reopened all 50 committed rollouts, recomputed official success and target error from the original branch reference states, rebuilt method aggregates, and checked all 4,510 candidate rows grouped into 820 decisions. Every decision had exactly one saved minimum-score winner, no source action appeared in a deployable candidate set, and the audit found zero discrepancies.

Completion-time disk space remained approximately 850 GB. EXP_G3 occupies approximately 61 MB.

## Artifacts

- Runner/auditor: `scripts/experiments/run_exp_g3_long_horizon.py`, `scripts/experiments/audit_exp_g3.py`
- Exact metadata/environment: `experiments/EXP_G3/run_metadata.json`, `environment.json`
- Branch and horizon definition: `experiments/EXP_G3/dataset_manifest.json`
- Candidate and case logs: `experiments/EXP_G3/candidate_interventions.jsonl`, `case_metrics.jsonl`
- Step-level committed evidence: `experiments/EXP_G3/rollouts/*.npz`
- Aggregate and audit: `experiments/EXP_G3/metrics.json`, `audit.json`

## Claim boundary and next decision

EXP_G3 directly falsifies the idea that short-horizon oracle state tracking with the existing F1/F2 proposal portfolio is sufficient for atomic manipulation. It does not falsify all latent control. The current latent is action-only and the controller does not infer object-contact phase from observation/state.

The next experiment must introduce a new model family rather than tune the shooting weights. EXP_G4 will fit closed-loop per-task state-conditioned behavior policies from train episodes, compare nearest-state retrieval, state-only residual imitation, and state-plus-action-latent residual imitation, and execute them over long development continuations. This tests whether physical-state conditioning fixes the missing long-horizon phase information and whether the learned latent adds value beyond state/action control.
