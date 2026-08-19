# EXP_G24 Report: Native-Action Visual Transition Coordinates

## Conclusion

**NOT SUPPORTED.** The aligned visual-transition latent completed 9/10 ordered tasks, but the unaligned visual latent and the capacity-matched direct visual world model also completed 9/10. Because ties explicitly do not establish a latent contribution, the action/visual-delta alignment hypothesis failed. The experiment nevertheless produced a useful system result: visual predictive ranking can retain native pi0.5 action bytes, reach 9/10, and outperform physical shooting's 8/10 while executing no candidate branches online. The evidence supports native-action visual prediction, not a necessary latent bottleneck.

## Interface repair and canonical data construction

The first G24 collection attempt exposed a real replay defect before producing usable branch data. Legacy snapshots preserved the operational-space controller's integration flag but omitted derived fields such as end-effector pose, Jacobians, velocities, and mass matrix. After restoring a snapshot, the first action could therefore consume controller state left by an earlier rollout. This also means G23's prospective decoded-action failures remain evidence that those controllers were unstable, but its matched-state comparison was not clean at the first restored action and should not be used as the strongest causal comparison.

G24 fixed the underlying checkpoint implementation in `src/pglt/libero/snapshot.py`: new snapshots preserve the derived controller fields, while legacy snapshots rebuild them from restored MuJoCo state and then restore integration fields once more. A two-environment restoration test recovered identical canonical end-effector state and zero site error. The failed original directory was retained as `experiments/EXP_G24`; no EXP ID was consumed by that pre-experiment gate.

The canonical retry then replayed the ten G18 initial conditions, queried fresh native pi0.5 proposals, and physically executed every candidate from a complete pre-decision simulator/controller checkpoint. It generated 1,242 real candidate transitions in 414 matched three-candidate groups, executing 6,265 candidate action steps. Each record stores the exact native action provenance, compact agent and wrist images before and after intervention, physical displacement, realized progress, success, and checkpoint. The collection controller itself completed 8/10 ordered tasks.

## Trained mechanisms and prospective evaluation

Ten leave-one-attempt-out folds trained a dual-view image encoder and four predictors per fold, for ten encoders and forty predictor checkpoints:

1. an action bottleneck aligned to realized visual change;
2. an otherwise matched unaligned visual latent;
3. a direct dual-view state/action visual world model;
4. a direct low-dimensional state/action world model.

The prospective evaluation used all ten canonical perturbed snapshots and autonomous state-window F3. Every learned ranker scored matched fresh proposals and committed the selected native pi0.5 bytes unchanged. Shuffled action-latent alignment isolated whether learned correspondence mattered; physical shooting executed candidates from restored checkpoints; single raw pi0.5 and initial-observation open loop supplied controller baselines.

## Results

| Method | Ordered success | Lift complete | Endpoint error | Jerk | Candidate steps |
|---|---:|---:|---:|---:|---:|
| aligned visual-transition latent | **9/10** | 10/10 | 0.05599 | 0.07949 | 0 |
| unaligned visual latent | **9/10** | 10/10 | 0.05058 | **0.07894** | 0 |
| direct visual world model | **9/10** | 10/10 | **0.04807** | 0.08073 | 0 |
| direct state world model | 7/10 | 10/10 | 0.05313 | 0.08221 | 0 |
| shuffled action latent | 6/10 | 10/10 | 0.05898 | 0.08449 | 0 |
| physical matched-branch shooting | 8/10 | 10/10 | 0.08534 | 0.08641 | 5,901 |
| single raw pi0.5 | 8/10 | 10/10 | 0.06050 | 0.08188 | 0 |
| initial-observation open loop | 0/10 | 0/10 | 1.09703 | 0.11393 | 0 |

Held-out offline ranking also does not isolate the proposed bottleneck. Aligned achieved ranking accuracy 0.6342 and mean regret 0.00610; unaligned achieved 0.6297/0.01205; direct visual achieved 0.6003/0.01209; direct state achieved 0.6118/0.00678. Aligned has lower regret than the other visual variants, but this did not translate into a strict prospective success gain. In closed-loop execution, mean aligned visual-latent residual was 0.14906 and physical residual was 0.01311. The corresponding values were 0.15021/0.01148 for unaligned and 0.15133/0.01016 for direct visual. These nearly indistinguishable visual residuals explain why the alignment mechanism cannot be credited for the 9/10 outcome.

The shuffled latent falling to 6/10 shows that learned action/outcome correspondence matters in this model family. However, direct visual prediction obtains the same 9/10 without the bottleneck, and the unaligned model does too. The scientifically justified winner is therefore the simpler native-action visual predictive family, with no evidence that aligned coordinates are necessary.

## Audit and artifacts

The independent audit passed with zero failures. It physically replayed all 1,242 branches using the original float64 proposal actions, checked all 414 complete controller checkpoints, recomputed all 40 fold models, 2,148 prospective predictions and residuals, verified 3,000 native proposal commits and all 80 rollout chains, and rebuilt the eight method aggregates. Physical states, action sequences, realized scores, and success labels matched exactly. Re-rendering the identical physical states under EGL had maximum mean RGB error 0.5526/255 and maximum 99.9-percentile error 5/255; sparse antialiased boundary pixels reached 64/255, so image replay is gated by the documented mean and tail criteria rather than a single edge pixel.

The two failed audit attempts were retained. The first demonstrated that down-casting proposal controls to float32 changes exact branch replay; the second demonstrated sparse EGL edge-pixel variation despite exact physical replay. The passing audit uses original float64 controls and preserves strict physical equality.

- `experiments/EXP_G24_retry1/visual_branch_dataset.npz`
- `experiments/EXP_G24_retry1/visual_branch_manifest.json`
- `experiments/EXP_G24_retry1/decision_checkpoints/`
- `experiments/EXP_G24_retry1/image_encoders/`, `fold_models/`, and `model_selection.json`
- `experiments/EXP_G24_retry1/rollouts/`, `case_metrics.jsonl`, `metrics.json`, and `audit.json`
- `experiments/EXP_G24_retry1/audit_failed_float32_pixel_identity.json`
- `experiments/EXP_G24_retry1/audit_failed_sparse_egl_edges.json`
- `scripts/experiments/collect_exp_g24_canonical_visual_branches.py`
- `scripts/experiments/run_exp_g24_visual_transition_latent.py`
- `scripts/experiments/audit_exp_g24.py`

The post-EXP disk check left 850 GB free.

## Consequence for the system

G24 improves the best native-action predictive controller to 9/10 and rejects visual alignment as the source of that gain. G25 will move the latent to a genuinely different role: temporal policy-side memory over realized intervention history. It will test whether a recurrent aligned history can resolve errors that a memoryless visual ranker cannot, against a same-capacity raw-action recurrent controller and the strongest G24 baselines. Native action bytes will remain untouched.
