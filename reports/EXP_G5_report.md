# EXP_G5 Report: Recurrent Closed-Loop Sequence Policies

## Conclusion

**NOT SUPPORTED: recurrent memory did not improve official completion over the G4 feed-forward policies.** Both state-only and state-plus-latent GRUs completed 0/10 tasks. The latent GRU reduced mean failure error, but this diagnostic improvement cannot substitute for completion.

## Experiment

EXP_G5 trained 20 task-local GRUs on complete Wave19 train trajectories: one state-only and one matched state-plus-latent model per task. Each model used hidden size 128, 64-step truncated sequences, 25 epochs, and the same optimizer/output normalization. At evaluation, the hidden state was warmed using only the saved physical/action history before the certified branch. After branching, recurrent updates consumed only realized simulator state, the previous executed control, elapsed time, and—only for the latent model—the rolling executed-action latent.

The exact ten G3/G4 development checkpoints and complete remaining horizons were retained. No test episode or future source action entered deployment.

## Results

| Method | Success | Mean final error ↓ | Mean jerk ↓ |
|---|---:|---:|---:|
| G4 state MLP | 1/10 | 0.468286 | 0.231406 |
| G4 state+latent MLP | **2/10** | 0.443123 | 0.225582 |
| state GRU | 0/10 | 0.342602 | **0.051310** |
| state+latent GRU | 0/10 | **0.272354** | 0.100289 |

The GRUs were much smoother and ended closer to the source target, but neither produced an official completion. Recurrent memory is therefore not supported as the next controller. The latent GRU's lower failure error is useful diagnostic evidence that action history affects trajectory tracking, but the registered latent-contribution condition required higher success than state-only GRU and was not met.

The result suggests that the dominant problem is not absent temporal memory alone. Both sequence policies remain trained only on expert-state distributions and receive no corrective examples after their own errors move the robot into unseen states.

## Audit and artifacts

The independent audit loaded all 20 GRU checkpoints, reopened all 20 new long-horizon rollouts, recomputed success and physical target error, and found zero discrepancies. EXP_G5 occupies approximately 19 MB; approximately 850 GB remained free after completion.

- Runner/auditor: `scripts/experiments/run_exp_g5_recurrent_policy.py`, `scripts/experiments/audit_exp_g5.py`
- Models/training: `experiments/EXP_G5/checkpoints/*.pt`, `training_summary.json`
- Metadata: `experiments/EXP_G5/run_metadata.json`, `environment.json`
- Rollouts/results: `experiments/EXP_G5/rollouts/*.npz`, `case_metrics.jsonl`, `metrics.json`, `audit.json`

## Next decision

G6 will address distribution shift directly through train-only closed-loop data aggregation. It will roll the current state-plus-latent MLP from certified train branches, label the learner-visited states with the corresponding demonstrated source control, refit an aggregated policy, and evaluate it without oracle actions on the unchanged development checkpoints.
