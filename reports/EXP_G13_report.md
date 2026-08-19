# EXP_G13 Report: Latent-Support F2 and Frozen Test Ablations

## Conclusion

**NOT SUPPORTED: latent-support F2 did not generalize to frozen test and the integrated system did not beat key simple baselines.** Development selected λ=0.05 at 4/5 versus λ=0 at 2/5, but frozen test reversed the result: latent support achieved 2/5 while latent-disabled, single-proposal F2, and F3-disabled future-visible control each achieved 3/5.

The full system did outperform restart (1/5) and open loop (0/5), and teacher replay reached 5/5, confirming all test starts are solvable. However, the final action-latent contribution and required strict ablation gains are unmet. No final success claim is allowed.

## Train-only support and development selection

Fourteen task-5 training episodes provided 1,166 pre-lift and 1,618 post-lift frozen 32-D action latents. Candidate continuations were scored by realized state value minus λ times nearest normalized latent-support distance for the protected phase.

All four λ values were executed on all five development starts with the frozen state-window F3:

| λ | Success | Final error ↓ | Candidate steps |
|---:|---:|---:|---:|
| 0 | 2/5 | 0.053867 | 1,718 |
| **0.05** | **4/5** | 0.049972 | 1,662 |
| 0.1 | 2/5 | **0.049908** | 1,716 |
| 0.2 | 2/5 | 0.063652 | 1,712 |

The preregistered success-first rule selected λ=0.05. The exact controller, F3, prompt, and λ were written to `frozen_system_manifest.json` before test artifacts were opened.

## One-time frozen test

Five task-5 test episodes were then evaluated once. Every baseline actually executed in LIBERO; open loop generated its complete sequence from repeated initial-observation π0.5 queries before execution, and teacher replay physically replayed archived successful controls.

| Test method | Success | Final error ↓ | Candidate steps | Interpretation |
|---|---:|---:|---:|---|
| full latent support | 2/5 | 0.105617 | 1,793 | selected system |
| latent disabled | **3/5** | 0.108828 | 1,752 | latent ablation wins success |
| F2 disabled, single π0.5 | **3/5** | 0.091282 | 0 | simple controller wins |
| F3 disabled, future-visible | **3/5** | **0.075236** | 2,034 | protected F3 gain does not transfer |
| restart after switch | 1/5 | 0.429226 | 1,928 | realized-state continuation matters |
| open loop from initial observation | 0/5 | 0.889056 | 0 | feedback is essential in general |
| teacher replay | 5/5 | 0.000239 | 0 | non-deployable upper bound |

The development/test reversal is central evidence. Latent support selected different proposals and doubled development success, but those choices overfit the five development starts. The state-only causal selector also failed to beat single sampling on test, so added causal computation is not automatically beneficial. Restart and open-loop failures still support physical retargeting and feedback relative to those baselines, but not the entire proposed architecture.

## Audit and artifacts

The independent audit verified train-only support, finite support statistics, the development winner, freeze-before-test timestamps, 20 development and 35 test rollouts, proposal-to-commit chains, teacher replay prefixes, open-loop raw proposal provenance, and all metrics. It passed without failures.

EXP_G13 occupies approximately 9.3 MB and left 849 GB free.

- `scripts/experiments/run_exp_g13_test_ablations.py`
- `scripts/experiments/audit_exp_g13.py`
- `experiments/EXP_G13/latent_support.npz`, `latent_support_manifest.json`
- `experiments/EXP_G13/development_rollouts/`, `development_selection.json`
- `experiments/EXP_G13/frozen_system_manifest.json`, `test_evaluation_manifest.json`
- `experiments/EXP_G13/test_rollouts/`, `test_case_metrics.jsonl`
- `experiments/EXP_G13/metrics.json`, `audit.json`, `run_metadata.json`

## Next decision

The frozen representation’s latent geometry is rejected as a generalizable support metric. EXP_G14 will learn a compact outcome-grounded action latent supervised by phase and future realized book/eef displacement, then compare matched state and new-latent temporal F3 systems on untouched official task-5 initial states that were never in Wave19.
