# EXP_G20 Report: Recovery-Trigger Gating

## Conclusion

**NOT SUPPORTED.** The latent recovery trigger completed 8/10 new causal rollouts. It beat the direct-state gate at 7/10, but tied the joint gate, never-return, and always-return controls at 8/10, while having worse endpoint error than all three. The adaptive-recovery criterion is false because no learned deployable gate strictly beat both fixed policies. The saved G19-outcome oracle reached 9/10, showing some trigger signal exists, but stochastic fresh continuation means it is not a deterministic upper bound.

## New mechanism and data split

G20 converted G19's ten audited paired interventions into a leave-one-attempt-out recovery decision problem. Each fold trained only on the other nine attempts. The label used task success first and endpoint error second to decide whether direct-state physical return had beaten no-return in G19. Three logistic gates used respectively the frozen 16-D action-progress latent, eight direct physical/disturbance features, or their joint representation. Fold membership, normalization, weights, probabilities, and held-out decisions were frozen before new rollout.

Every evaluated method then executed a fresh closed-loop continuation from the exact G19 disturbed snapshot. A positive trigger ran the audited OSC return controller toward the direct-state-selected checkpoint; a negative trigger continued immediately. Thus G20 evaluated the gate through `decision -> optional executed return -> realized state -> fresh pi0.5 feedback`, not by selecting old outcomes.

## Results

| Method | Success | Triggered returns | Recovery completion | Endpoint error down |
|---|---:|---:|---:|---:|
| latent gate | 8/10 | 4 | 4/4 | 0.088359 |
| direct-state gate | 7/10 | 3 | 3/3 | **0.061239** |
| joint latent+state gate | 8/10 | 6 | 6/6 | 0.080464 |
| never return | 8/10 | 0 | n/a | 0.073351 |
| always return | 8/10 | 10 | 10/10 | 0.069189 |
| saved-outcome oracle trigger | **9/10** | 2 | 2/2 | **0.055595** |

The latent-only gate's one-success advantage over state-only is insufficient: it had no success advantage over the two fixed policies or joint gate. In addition, its error was highest among all methods except none. The proper conclusion is that this small leave-one-attempt-out latent gate did not add control value. Together with G19, the evidence says checkpoint return should be removed from the primary system for this disturbance regime.

## Audit and artifacts

The independent audit passed. It rebuilt G19 paired labels/features, refit all 30 fold models and reproduced their weights/probabilities, checked all 60 trigger decisions and continuation chains, verified 25 physical returns, and reconstructed all six aggregate rows.

- `scripts/experiments/run_exp_g20_recovery_trigger.py`
- `scripts/experiments/audit_exp_g20.py`
- `experiments/EXP_G20/gate_folds.json`
- `experiments/EXP_G20/frozen_system_manifest.json`
- `experiments/EXP_G20/recovered_snapshots/`, `rollouts/`
- `experiments/EXP_G20/case_metrics.jsonl`, `metrics.json`, `audit.json`

The post-EXP disk check left 850 GB free; G20 occupies about 20 MB.

## Next decision

G21 removes checkpoint return from the primary controller. It will learn a causal-regret representation from thousands of matched candidate branches and use it to decide when expensive multi-proposal F2 is warranted. This is distinct from G18: the latent will not rank candidates, but will control whether to branch at all. It will be compared with a direct regret classifier, always-shooting, and single-proposal control on new executed rollouts, with both task success and intervention efficiency measured.
