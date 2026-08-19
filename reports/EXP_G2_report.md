# EXP_G2 Report: Causal F2 Isolation under Oracle Current-Action Targets

## Conclusion

**SUPPORTED, within the oracle-boundary local-control scope: causal latent-augmented shooting reduced realized 16-step physical target error relative to open-loop F2, receding F1, and direct state/action shooting.** This is not yet evidence of atomic-action completion: all evaluated methods had zero official task success over the short horizon.

## Hypothesis and new mechanism

EXP_G2 asked whether learned F1/F2 proposals become useful when they are evaluated through actual simulator interventions rather than offline latent prediction. It introduced a receding matched-state shooting loop. At each four-step decision boundary, candidate proposals were executed from an identical full checkpoint, their realized physical endpoints were scored, the state was restored, and the best proposal was committed. The next decision used the committed realized state and an action-history latent re-encoded after execution.

This is new causal evidence relative to EXP_G1, which only established restoration and intervention sensitivity. EXP_G2 loaded the independently trained Wave20 LIBERO semantic/F1/F2 checkpoints and compared them inside an executed controller.

## Data and protocol

- Ten development episodes, one per LIBERO-10 task; no frozen test episode was opened.
- One certified `branch_025` checkpoint per episode.
- Sixteen committed controller steps per method, with four-step receding prefixes.
- Oracle scope: the recorded source physical continuation supplied the same-action target state and completion boundary. Future source actions were excluded from every deployable candidate set and used only by the separately labeled `source_oracle` reference.
- Causal score: normalized realized end-effector error, realized body-position error, and action discontinuity; `causal_latent` additionally included latent support radius. The normalization scales came from EXP_G1 development intervention medians.

Compared methods were `source_oracle`, `open_loop_F1`, `open_loop_F2`, `receding_F1`, `receding_F2`, `causal_state`, and `causal_latent`. `causal_state` shot only direct/damped/hold/norm-matched-random action candidates. `causal_latent` added F1, F2, and current-latent copy proposals.

## Executed evidence

The completed experiment contains 1,120 committed controller steps and 1,760 candidate-intervention steps, for 2,880 actual simulator/controller steps. Candidate rollouts were not treated as model predictions: each was executed through LIBERO from a restored checkpoint and produced a realized state before scoring.

| Method | Mean final physical error ↓ | Mean physical progress ↑ | Mean jerk ↓ | Success |
|---|---:|---:|---:|---:|
| source oracle | 0.000098 | 0.273499 | 0.056512 | 0% |
| open-loop F1 | 0.031905 | 0.011735 | 0.078772 | 0% |
| open-loop F2 | 0.033121 | 0.032470 | 0.079859 | 0% |
| receding F1 | 0.036411 | 0.029919 | 0.091153 | 0% |
| receding F2 | 0.040505 | 0.028199 | 0.094684 | 0% |
| causal state | 0.049648 | 0.021921 | 0.033996 | 0% |
| **causal latent** | **0.022001** | **0.055998** | 0.047402 | 0% |

The causal-latent controller was the best deployable method on the registered primary metric. Its 40 decisions selected direct controls 19 times, F1 eight times, F2 six times, latent copy three times, and damped/hold controls four times. Thus 17/40 selections came from learned or copied latent proposals. The gain is not explained by blindly executing F2: receding F2 had the worst learned-controller endpoint error, and open-loop F1 beat open-loop F2.

## Interpretation

Direct evidence supports the mechanism-level statement that latent proposal diversity can add value to a physical-feedback shooting controller. It does not support the stronger statement that the historical F2 refinement is individually superior to F1. F1 was selected more frequently than F2, open-loop F1 beat open-loop F2, and receding F1 beat receding F2. The best controller should therefore retain a proposal portfolio rather than granting F2 exclusive authority.

The comparison between `causal_latent` and `causal_state` shows value beyond the tested direct candidate set, but it is not a universal proof against all state-space controllers. The latent method had a larger candidate set as part of the registered mechanism. Later ablations must distinguish proposal diversity, latent support, and learned F1/F2 content.

All official task-success rates were zero because the 16-step horizon ended far before completion. Consequently EXP_G2 supports only local oracle-target progress. It does not establish atomic completion, learned F3, current-action protection, retargeting, or ordered composition.

## Failures retained during execution

Two incomplete implementation attempts are preserved under `experiments/debug/`: one exposed Python 3.8's lack of `str.removeprefix`, and one exposed JSON serialization of `numpy.bool_`. Neither attempt was counted as an EXP. Their partial artifacts were not included in the formal aggregate.

## Artifacts and reproduction

- Runner: `scripts/experiments/run_exp_g2_causal_f2.py`
- Auditor: `scripts/experiments/audit_exp_g2.py`
- Run/environment records: `experiments/EXP_G2/run_metadata.json`, `environment.json`
- Case membership: `experiments/EXP_G2/dataset_manifest.json`
- Candidate interventions: `experiments/EXP_G2/candidate_interventions.jsonl`
- Per-case metrics: `experiments/EXP_G2/case_metrics.jsonl`
- Step-level committed rollouts: `experiments/EXP_G2/rollouts/*.npz`
- Aggregate metrics and audit: `experiments/EXP_G2/metrics.json`, `audit.json`

```bash
PYTHONPATH=src:/home/jinjaguo/LIBERO \
MUJOCO_GL=egl NUMBA_CACHE_DIR=/tmp/pglt_numba_cache \
MPLCONFIGDIR=/tmp/pglt_matplotlib \
/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/run_exp_g2_causal_f2.py \
  --output experiments/EXP_G2 --branches-per-task 1 \
  --horizon 16 --prefix 4 --device cpu

/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/audit_exp_g2.py --experiment experiments/EXP_G2 --tolerance 1e-12
```

The independent audit reopened all 70 committed rollout archives, recomputed endpoint errors from the original branch reference states, rebuilt the method aggregates and winner, and checked all 440 candidate records and 80 selection groups. It found zero discrepancies. Completion-time free space remained approximately 850 GB; EXP_G2 occupies approximately 11 MB.

## Next decision

The primary bottleneck is now long-horizon current-action completion. EXP_G3 will retain oracle completion boundaries, start from the latest available certified development branches, and run each method over the complete remaining source-continuation horizon. This tests whether the local causal-latent advantage survives contact-rich long-horizon execution and produces official task completion.
