# EXP_G8 Report: Dual-Camera Visual Chunk Policies

## Conclusion

**NOT SUPPORTED: the lightweight dual-camera visual policies did not exceed the G4 state-policy baseline, and the learned action latent hurt the matched visual controller.** Both visual variants completed 0/10 late-checkpoint tasks, versus 2/10 for G4 state+latent MLP. EXP_G8 therefore does not advance atomic execution reliability and does not satisfy any integrated `lift -> place` acceptance condition.

## Scientific question and new mechanism

EXP_G4–G7 showed a 2/10 ceiling for policies driven only by robot/body state and action history. EXP_G8 tested whether the missing information was object identity and contact appearance. It introduced a shared, task-conditioned dual-camera CNN trained at the source π0.5 policy-issue timebase. The matched ablation used the same images, robot state, task identity, optimizer, target chunks, and deployment loop, differing only by the addition of the frozen rolling 32-D action latent.

## Data and training

Training used only the Wave19 training split: 140 successful episodes and 7,402 policy-issue windows. Every window contained processed agent-view RGB, processed wrist RGB, the saved 8-D robot state, task identity, and a 10×7 postprocessed π0.5 action chunk. Saved 224×224 images were bilinearly reduced to 32×32. The latent variant encoded the preceding 16 actually executed controls with the frozen Wave19 representation.

Each policy used a shared three-layer CNN for the two views, concatenated the two image features with state and task one-hot features, and regressed a ten-control chunk. Both variants trained for 15 CPU epochs with AdamW at 1e-3. The final normalized training MSE was 0.254425 for visual state and 0.058320 for visual state+latent. The much smaller fitted loss of the latent model did not transfer to control success.

## Causal execution protocol

The evaluation reused the unchanged ten late development checkpoints from EXP_G3, one per task. At every decision the policy received the currently rendered agent and wrist images plus the physically realized robot state. It predicted ten controls, the first five were sign-normalized for the gripper and executed through LIBERO, new observations were captured, the action history/latent was updated, and the policy replanned. No source future image, state, action, or task-success boundary was available online.

This generated 20 new closed-loop rollouts. Tasks 0–4 retained their development role and tasks 5–9 were reported separately as confirmation; neither half contained a success.

| Method | Success | Tasks 0–4 | Tasks 5–9 | Mean final error ↓ | Executed-action jerk ↓ |
|---|---:|---:|---:|---:|---:|
| G4 state+latent one-step MLP | **2/10** | — | — | 0.443123 | 0.225582 |
| visual state | 0/10 | 0/5 | 0/5 | **0.384547** | 0.162618 |
| visual state+latent | 0/10 | 0/5 | 0/5 | 0.658027 | **0.049623** |

The pure visual-state controller reduced mean endpoint error relative to G4 but did not cross any official task-success predicate. The latent variant was smoother but had substantially worse endpoint error. Thus neither “visual control exceeds 2/10” nor “latent improves the matched visual policy” is supported. The train/control gap is direct evidence that low imitation loss on source issue windows is not sufficient for recovery from late-checkpoint distribution shift.

## Audit and reproducibility

The independent audit loaded both checkpoints, verified that all 140 contributing episodes belonged to the declared training split, reopened all 20 rollout archives, and exactly recomputed success and terminal target error. It passed with no failures. EXP_G8 occupies approximately 3.4 MB, and 849 GB remained available after completion.

Reproduction command:

```bash
PYTHONPATH=src:/home/jinjaguo/LIBERO MUJOCO_GL=egl \
NUMBA_CACHE_DIR=/tmp/pglt_numba_cache MPLCONFIGDIR=/tmp/pglt_matplotlib \
/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/run_exp_g8_visual_policy.py \
  --output experiments/EXP_G8 --image-size 32 --epochs 15 \
  --replan-steps 5 --device cpu
```

Key evidence:

- `scripts/experiments/run_exp_g8_visual_policy.py`
- `scripts/experiments/audit_exp_g8.py`
- `experiments/EXP_G8/training_window_manifest.json`
- `experiments/EXP_G8/visual_state.pt`, `visual_state_latent.pt`
- `experiments/EXP_G8/rollouts/`, `case_metrics.jsonl`
- `experiments/EXP_G8/metrics.json`, `audit.json`, `run_metadata.json`

## Next decision

The small supervised CNN is rejected as the next low-level controller. EXP_G9 will use the locally available official π0.5 checkpoint directly in the loop and test whether multiple stochastic π0.5 proposals, frozen-latent proposal ranking, and causal simulator-feedback ranking can recover robust atomic execution. This changes both the model family and the intervention mechanism instead of tuning EXP_G8.
