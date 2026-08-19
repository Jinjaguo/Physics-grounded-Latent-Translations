# EXP_G12 Report: Learned Temporal F3 End-to-End Integration

## Conclusion

**SUPPORTED: the state-window temporal F3 autonomously switched a protected `lift -> contextual place` system and achieved 4/5 official success.** It exceeded train-median fixed switching at 3/5 and future-visible full-prompt control at 1/5, while matching the oracle F3’s 4/5. It switched on every start, had no premature switches, retained lift 5/5, and achieved the lowest endpoint error, 0.04170.

**NOT SUPPORTED: the learned action latent improved F3.** The matched state+latent temporal model had perfect train classification but only 2/5 end-to-end success, compared with 4/5 for state only. This is a direct train/control generalization failure and the latent should be removed from F3.

EXP_G12 demonstrates the core autonomous two-action architecture on development starts: current action hidden from the future prompt, causal candidate execution and realized feedback at every F2 decision, autonomous learned switching, and place retargeting from the physical state reached by lift. Final acceptance is not yet met because the integrated latent contribution is negative and frozen held-out/test ablations remain.

## Train-only temporal dataset

Fourteen task-5 training episodes produced 562 policy-issue windows. Raw issue steps begin at simulator step 10 and were aligned to certified action time by subtracting the recorded ten stabilization steps. Labels become positive at the third consecutive step satisfying the 4 cm lift predicate. The median training trigger was action step 83.

Both GRUs used the latest eight issue features with left padding and an explicit mask. The state model received the saved 8-D policy state plus mask; the matched latent model additionally received the frozen 32-D rolling action latent. There were 324 positive and 238 negative windows.

| F3 train model | Final BCE | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| state window | 0.020684 | 0.9875 | 0.9847 | 0.9938 |
| state+latent window | **0.003055** | **1.0000** | **1.0000** | **1.0000** |

The latent model’s perfect training result did not predict closed-loop utility.

## End-to-end causal protocol

All methods started before policy control on the same five task-5 development episodes. The low-level controller remained G9’s two-proposal causal state-value F2: every proposal prefix was executed from the same decision snapshot, scored using realized state, then the simulator was restored and the winner committed. All protected methods exposed only `pick up the black book` before switching and G11’s winning completed-lift contextual place prompt after switching.

The learned F3 methods used only recent policy robot state and optional action latent online. Book height was logged only after execution to compute retrospective switch error; it did not affect their switches.

| Method | Success | Switch | Premature | Mean |switch error| | Final error ↓ | Candidate steps |
|---|---:|---:|---:|---:|---:|---:|
| oracle F3 | 4/5 | 5/5 | 0 | 0.0 | 0.051542 | 1,665 |
| train-median fixed F3 | 3/5 | 5/5 | 1 | **5.6** | 0.045772 | 1,716 |
| **state-window learned F3** | **4/5** | 5/5 | 0 | 7.6 | **0.041700** | 1,692 |
| state+latent learned F3 | 2/5 | 5/5 | 0 | 8.4 | 0.051679 | 1,740 |
| future-visible full prompt | 1/5 | 0/5 | 0 | — | 0.067023 | 1,734 |

The learned state F3’s mean switch error was not the smallest, yet its downstream result was best. This shows that exact imitation of the height boundary is not the only useful switching criterion. Future-visible control’s 1/5 result versus protected learned F3’s 4/5 supplies the first strong current-action protection evidence.

## Audit and artifacts

The independent audit verified all fitting episodes were training-only, loaded two temporal checkpoints, reconstructed 25 retrospective lift/switch traces, checked the 0.5 threshold and two-positive persistence on all ten learned-F3 rollouts, and exactly matched every selected proposal prefix to committed simulator actions. It reproduced all aggregates with no failures.

EXP_G12 occupies approximately 6.0 MB and left 849 GB free.

- `scripts/experiments/run_exp_g12_learned_f3.py`
- `scripts/experiments/audit_exp_g12.py`
- `experiments/EXP_G12/f3_dataset_manifest.json`, `training_summary.json`
- `experiments/EXP_G12/temporal_f3_state.pt`, `temporal_f3_state_latent.pt`
- `experiments/EXP_G12/rollouts/`, `case_metrics.jsonl`
- `experiments/EXP_G12/metrics.json`, `audit.json`, `run_metadata.json`

## Next decision

The integrated state-F3 architecture is frozen. EXP_G13 will relocate the action latent from F3 into a proposal-support constraint, select its strength in a single development sweep, then open task-5 test episodes once for confirmation and run the full latent-disabled, F2-disabled, F3-disabled, restart, open-loop, and teacher-replay ablations required for final acceptance.
