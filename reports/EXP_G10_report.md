# EXP_G10 Report: Oracle-Switched Lift-to-Place Composition

## Conclusion

**NOT SUPPORTED: oracle-switched hard gating did not produce reliable `lift -> place` completion and did not improve current-action protection.** All four mechanisms completed and retained lift on all five matched development starts, so hiding the future place action offered no lift advantage. The hard-gated realized-state system switched 5/5 times but completed place 0/5; the future-visible full prompt and place-only prompt each completed 1/5.

**SUPPORTED ONLY AS A STATE-DEPENDENCE ABLATION: continuing from the realized lifted state was materially better than restarting before place.** Realized retarget and restart both had 0/5 official success, but mean final target error was 0.0816 versus 0.5023 and restart deliberately lost the lift in 5/5 cases. This is not enough to claim successful composition.

## New dataset and oracle boundary

EXP_G10 used all five Wave19 development episodes for LIBERO task 5 and did not open the test split. Every method began from the exact snapshot after the episode’s ten stabilization steps, before any source-policy control. MuJoCo inspection verified body 23 as `black_book_1_main` in the actual task model.

The explicit oracle F3 predicate required the book to remain at least 4 cm above its initial height for three consecutive simulator steps while the final task predicate was still false. On the five archived successful development trajectories, this boundary occurred at action indices 77–90, well before official success at 169–184. Thus the predicate isolates a real lift phase, but remains simulator-ground-truth oracle logic and is not autonomous learned F3.

## Methods and execution

All methods reused G9’s two-proposal causal state-value F2. At every decision, two fresh π0.5 prefixes were physically executed from the same exact decision snapshot, their realized states were scored with the frozen task-5 training-only value, the simulator was restored, and the selected prefix was committed.

- `future_visible_full_prompt`: original composite prompt throughout.
- `hard_gate_realized_retarget`: lift-only prompt until oracle boundary, then place-only prompt from the physically reached state.
- `hard_gate_restart_place`: identical protected lift and switch, but restore the initial state before place.
- `future_only_place`: place-only prompt from the initial state.

The first two implementation attempts failed before saving a rollout because the fixed action encoder requires a left-padded 16-step window at early time steps. Both directories were preserved under `experiments/debug/`; the corrected helper was directly verified at history lengths 0, 1, 15, 16, and 17 before formal execution.

| Method | Lift | Switch | Official success | Lift loss | Final error ↓ | Jerk ↓ | Candidate steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| future-visible full | 5/5 | 0/5 | **1/5** | 0/5 | **0.047353** | 0.070406 | 1,750 |
| hard gate, realized retarget | 5/5 | 5/5 | 0/5 | 0/5 | 0.081581 | **0.067070** | 1,786 |
| hard gate, restart place | 5/5 | 5/5 | 0/5 | 5/5 | 0.502284 | 0.112504 | 1,768 |
| future-only place | 5/5 | 0/5 | **1/5** | 0/5 | 0.072503 | 0.074682 | 1,722 |

The surprising 5/5 lift rate of the place-only prompt shows that π0.5’s natural-language behavior is not cleanly atomic: it often performs prerequisite pickup implicitly. Hard prompt gating therefore does not itself create the desired modular action interface. The 0/5 place result after a clean hard switch identifies post-lift place control and prompt/value mismatch as the next bottleneck.

## Audit and artifacts

The independent audit verified development-only membership, the body mapping, all five archived lift boundaries, prompt transition traces, restart-state book height, every official success/error metric, and all 20 variable-length selected-proposal-to-committed-action chains. It passed with no failures.

EXP_G10 occupies approximately 5.2 MB, its two retained failed attempts occupy 28 KB each, and 849 GB remained free after completion.

- `scripts/experiments/run_exp_g10_lift_place.py`
- `scripts/experiments/audit_exp_g10.py`
- `experiments/EXP_G10/composition_dataset_manifest.json`
- `experiments/EXP_G10/verified_body_mapping.json`
- `experiments/EXP_G10/archived_oracle_boundary_validation.json`
- `experiments/EXP_G10/rollouts/`, `case_metrics.jsonl`
- `experiments/EXP_G10/metrics.json`, `audit.json`, `run_metadata.json`

## Next decision

Learned F3 is premature because even oracle-switched place fails. EXP_G11 will generate one exact realized lift snapshot per matched start, then branch post-lift controllers from that identical state. It will test whether preserving task context in the prompt and training a post-lift-specific state or state+latent value can repair place control.
