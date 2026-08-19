# EXP_G23 Report: Joint Action-Bottleneck Predictive Control

## Conclusion

**NOT SUPPORTED.** The preregistered continuous joint action latent completed 4/10 ordered tasks. A capacity-matched direct action world model completed 8/10, physical matched-branch shooting completed 8/10, and the simplest single-proposal pi0.5 controller completed 9/10. The joint latent therefore failed every required comparison. Vector quantization was substantially worse at 0/10. The evidence says that replacing native pi0.5 control bytes with decoded action chunks is harmful in this interface, even when the action bottleneck is jointly trained on realized dynamics.

## New data and model family

G23 rebuilt candidate transitions from audited G15--G18 physical branches. It retained only decisions where all five controls of the executed prefix were actually run; 46 truncated tail decisions were excluded rather than treating unexecuted padding as supervision. The final dataset contains 7,188 candidate transitions in 2,642 decision groups. Every item links exact pre-decision EEF/object state and active phase, a 5x7 native pi0.5 control prefix, physically realized EEF/object displacement, success, a stage-specific realized progress score, source artifact, decision, candidate, and attempt.

Four families were trained in ten leave-one-attempt-out folds, producing 40 checkpoints:

1. a 32-code vector-quantized action bottleneck with executable decoder and joint transition/score/success heads;
2. an 8-D continuous action bottleneck with the same joint heads;
3. an action-only autoencoder whose transition head was fitted after freezing the encoder/decoder;
4. a direct state-plus-action world model without a bottleneck or decoder.

The continuous joint family was selected before prospective rollout because it beat VQ within the latent-family rule. Its held-out ranking accuracy was 0.7246 and displacement MAE 0.00334, versus VQ's 0.5428/0.00497. Importantly, the direct model was already better than the selected latent offline at 0.7325 ranking accuracy and 0.00278 displacement MAE. The frozen manifest records this selection before any G23 rollout outcome was read.

## Executed controller mechanisms

All eight methods used the ten matched G18 perturbed snapshots and autonomous state-window F3. Model-based methods queried native pi0.5 proposals but did not execute candidates for selection. VQ explored native codes and nearby codebook entries; continuous control performed five bounded gradient steps in latent space; both decoded the chosen latent to a real 5-step control prefix. Action-only used its frozen representation without joint encoder updates. Direct control ranked and executed untouched native proposals. `decoded_first` measured decoder damage without latent optimization. Physical shooting executed all candidates from matched checkpoints. Single raw pi0.5 and initial-observation open loop completed the controls.

Every decoded or native selected prefix was actually committed through LIBERO. The next decision consumed the physically realized observation. Thus the negative result is from causal closed-loop execution, not offline ranking or teacher-forced replay.

## Results

| Method | Ordered success | Lift complete | Endpoint error | Jerk | Candidate steps |
|---|---:|---:|---:|---:|---:|
| continuous joint latent MPC | 4/10 | 7/10 | 0.34922 | 0.11322 | 0 |
| VQ joint latent MPC | 0/10 | 0/10 | 0.87968 | 0.08140 | 0 |
| action-only latent | 2/10 | 6/10 | 0.40239 | 0.13293 | 0 |
| direct action world model | 8/10 | 10/10 | 0.07513 | 0.08044 | 0 |
| decoded first proposal | 5/10 | 7/10 | 0.31560 | 0.11765 | 0 |
| physical matched-branch shooting | 8/10 | 10/10 | **0.06065** | 0.08716 | 6,159 |
| single raw pi0.5 | **9/10** | 10/10 | 0.07064 | **0.07633** | 0 |
| initial-observation open loop | 0/10 | 1/10 | 1.10448 | 0.11336 | 0 |

Joint transition training did matter relative to the action-only latent: continuous reached 4 successes versus 2. It did not overcome action reconstruction and model-exploitation error. Across all decisions, continuous latent intervention changed its source native action by mean per-step L2 0.3224 and had mean realized transition residual 0.02058. Decoding without optimization still shifted actions by 0.2917 and reached only 5/10. VQ shifted actions by 0.7550 and had 0.22169 residual, explaining its complete failure. In contrast, the direct model changed no action bytes and had only 0.01131 transition residual.

The physical shooting result also confirms that more causal branch computation is not automatically better: it used 6,159 candidate-executed steps yet lost one task to single raw pi0.5. The strongest current integrated controller remains the simpler native single-proposal feedback plus autonomous state F3.

## Audit and artifacts

The independent audit passed. It rebuilt all 7,188 candidate records and 2,642 groups from source rollouts, validated exact membership and all 40 folds, recomputed 2,542 predictive decisions including VQ neighborhoods and continuous gradients, checked 3,364 proposal commits and 2,542 predicted-versus-realized residuals, validated all 80 rollout chains, and reconstructed all eight method aggregates.

- `scripts/experiments/run_exp_g23_joint_action_world_model.py`
- `scripts/experiments/audit_exp_g23.py`
- `experiments/EXP_G23/executed_prefix_transition_dataset.npz`
- `experiments/EXP_G23/transition_dataset_manifest.json`
- `experiments/EXP_G23/fold_models/`, `model_selection.json`, `frozen_system_manifest.json`
- `experiments/EXP_G23/rollouts/`, `case_metrics.jsonl`, `metrics.json`, `audit.json`

The post-EXP disk check left 850 GB free.

## Consequence for the system

G23 rejects a stronger version of the prior action-latent hypothesis: even joint dynamics training does not make a lossy decoded action coordinate preferable to native pi0.5 controls. G24 will preserve native action bytes and change the learned coordinate to an action-conditioned visual transition latent. It will collect real before/after images from restored candidate branches, align action latents to controllable visual change, and use that latent only to rank untouched proposals against direct visual/state world models.
