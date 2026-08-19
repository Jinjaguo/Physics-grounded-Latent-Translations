# EXP_G9 Report: Live π0.5 Proposal Control with Causal Feedback

## Conclusion

**SUPPORTED: the official π0.5 low-level family and executed state-value proposal selection broke the 2/10 learned-controller ceiling.** `pi05_causal_state_value` achieved 6/10 overall with 3/5 confirmation tasks, compared with 5/10 for a fresh single-sample π0.5 loop and 2/10 for G4 state+latent MLP. **NOT SUPPORTED: adding the frozen action latent to the matched causal value improved control.** It reduced success from 6/10 to 2/10.

This is the first deployable formulation in the G series to exceed both the learned state-policy baseline and the archived 5/10 source-reference success count on the same late checkpoints. It establishes a sufficiently strong atomic controller for the first explicit `lift -> place` composition experiment, but it does not itself test switching or satisfy the final system goal.

## New mechanism and data

EXP_G9 replaced the failed small CNN with the locally installed official 14 GB `pi05_libero` PyTorch checkpoint and its official normalization assets. At each decision, live dual-camera images, 8-D robot state, and the actual episode instruction were sent to the websocket policy. The server generated fresh stochastic 10×7 chunks; archived future controls were never issued online.

Twenty task-local progress values were trained from all 140 Wave19 training episodes: one state-only and one matched state+32-D-action-latent model per task. Targets were the normalized action index within successful training trajectories. The target phase was explicitly excluded from model inputs. Depending on task, each value used 2,784–5,847 training states. Although state+latent usually achieved lower training MSE, that fit advantage did not transfer to causal selection.

The first run failed before any rollout because object/body counts vary across LIBERO tasks, making a shared vector input ill-defined. That implementation failure was retained under `experiments/debug/EXP_G9_attempt1_variable_body_shapes/`; task-local matched values resolved the gate without consuming another EXP.

## Methods and causal protocol

All methods started from the same ten late EXP_G3 snapshots and replanned every five committed controls:

- `pi05_single`: one fresh π0.5 sample per decision.
- `pi05_open_latent_value`: two stochastic proposals; rank using the latent value without executing candidates.
- `pi05_causal_state_value`: restore the decision snapshot for each proposal, physically execute its five-step prefix, score the realized state using the train-only state value, restore, then commit the winner.
- `pi05_causal_state_latent_value`: identical causal candidate execution and selection, with the matched state+latent value.

Every committed step updated the rendered observation, physical state, action history, and latent before the next policy call. The two causal methods executed 2,938 and 2,875 candidate steps respectively in addition to their committed actions.

| Method | Success | Tasks 0–4 | Tasks 5–9 | Final error ↓ | Jerk ↓ | Candidate steps | Changed selection |
|---|---:|---:|---:|---:|---:|---:|---:|
| G4 state+latent MLP | 2/10 | — | — | 0.443123 | 0.225582 | 0 | — |
| archived source reference | 5/10 | — | — | **0.000785** | 0.116032 | 0 | — |
| π0.5 single | 5/10 | 2/5 | 3/5 | 0.080935 | 0.114856 | 0 | 0 |
| π0.5 open latent value | 4/10 | 3/5 | 1/5 | 0.123044 | 0.120574 | 0 | 158 |
| **π0.5 causal state value** | **6/10** | **3/5** | **3/5** | 0.047377 | **0.108942** | 2,938 | 136 |
| π0.5 causal state+latent value | 2/10 | 1/5 | 1/5 | 0.094099 | 0.117563 | 2,875 | 142 |

The causal state selector improved both success and error relative to single sampling. The action latent was not merely neutral: both open latent ranking and the matched causal latent value selected many non-first proposals but performed worse. The scientifically supported system at this point should therefore retain the latent for measurement/ablation but use state-value feedback for low-level proposal selection.

## Audit and reproducibility

The independent audit loaded all 20 value checkpoints, verified all 140 source episodes were in the training split, reopened all 40 rollouts, and exactly reconstructed success, terminal error, committed/candidate step counts, and selection counts. For every decision it verified that the saved selected proposal prefix exactly equals the saved committed actions. All 40 causal chains passed.

EXP_G9 occupies approximately 18 MB and left 849 GB free. The official GPU server used seed 190819; the controller command was:

```bash
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src \
MUJOCO_GL=egl NUMBA_CACHE_DIR=/tmp/pglt_numba_cache MPLCONFIGDIR=/tmp/pglt_matplotlib \
/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/run_exp_g9_pi05_causal_value.py \
  --output experiments/EXP_G9 --host localhost --port 8000 \
  --proposals 2 --replan-steps 5 --epochs 20 --device cpu
```

Key evidence:

- `scripts/experiments/run_exp_g9_pi05_causal_value.py`
- `scripts/experiments/audit_exp_g9.py`
- `experiments/EXP_G9/value_dataset_manifest.json`, `training_summary.json`
- `experiments/EXP_G9/progress_value_task*.pt`
- `experiments/EXP_G9/rollouts/`, `case_metrics.jsonl`
- `experiments/EXP_G9/metrics.json`, `audit.json`, `run_metadata.json`

## Next decision

Atomic execution is no longer the only bottleneck. EXP_G10 will decompose the real task-5 instruction into `lift black book -> place black book in caddy`, use ground-truth object elevation as oracle F3, and compare future-visible control, hard current-action gating with physically realized retargeting, and restart/place baselines under the G9 causal state-value controller.
