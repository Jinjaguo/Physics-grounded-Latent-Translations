# EXP_G25 Report: Temporal Policy-Side Latent Memory

## Conclusion

**NOT SUPPORTED.** The aligned latent-history GRU completed 7/10 ordered tasks. The exactly equal-parameter raw-history GRU completed 9/10, and the memoryless aligned visual scorer also completed 9/10. Shuffling or resetting latent history each decision both completed 7/10, tying the proposed recurrent latent. Temporal latent conditioning therefore failed the required representation and mechanism ablations. The useful result is simpler: raw causal history can support a strong native-action ranker, but the learned action latent does not explain the gain.

## Causal sequential construction

G25 transformed the audited G24 intervention dataset into chronological decision sequences. It retained 1,242 physically executed candidate branches in 414 three-candidate groups across ten attempts. At each decision the models could read only current state/proposals and prior selected transitions that had already been physically executed. Outcomes of uncommitted candidates remained training targets and never entered prospective history. Thus the recurrent state did not receive teacher-forced future observations or unexecuted branch outcomes.

For every leave-one-attempt-out fold, two GRU rankers were trained:

1. aligned history used learned 16-D action/visual-transition coordinates together with predicted and realized transition features;
2. raw history used the untouched 35-D native action prefix together with direct visual/physical transition features.

Both families padded their inputs to the same 128-D width, used the same 64-D GRU state and scoring network, and had exactly 49,665 trainable parameters. Ten folds produced twenty checkpoints. The held-out latent ranker obtained 0.6996 mean ranking accuracy and 0.01173 regret; raw history obtained 0.7256 accuracy and 0.01175 regret. The small regret reversal does not outweigh raw history's better decision accuracy or prospective success.

## Prospective mechanisms

Ninety fresh closed-loop rollouts were executed from the ten canonical G24 simulator/controller snapshots with autonomous state-window F3. Every learned method queried matched pi0.5 candidates, selected an index, committed the original action bytes, observed the realized image and physical transition, and only then updated its recurrent state. The comparisons included normal aligned history, equal-capacity raw history, memoryless aligned and direct-visual models, reverse-ordered causal latent history, a reset-at-every-decision latent ablation, physical branch shooting, single raw pi0.5, and initial-observation open loop.

## Results

| Method | Ordered success | Lift complete | Endpoint error | Jerk | Candidate steps |
|---|---:|---:|---:|---:|---:|
| aligned latent-history GRU | 7/10 | 10/10 | 0.08428 | **0.08004** | 0 |
| raw-history GRU | **9/10** | 10/10 | **0.04913** | 0.08086 | 0 |
| memoryless aligned visual | **9/10** | 10/10 | 0.05386 | 0.08267 | 0 |
| direct visual world model | 7/10 | 10/10 | 0.09855 | 0.08446 | 0 |
| shuffled latent history | 7/10 | 10/10 | 0.09038 | 0.08036 | 0 |
| reset latent history | 7/10 | 10/10 | 0.05930 | 0.08455 | 0 |
| physical matched-branch shooting | 7/10 | 10/10 | 0.07032 | 0.08190 | 6,643 |
| single raw pi0.5 | 7/10 | 10/10 | 0.09383 | 0.08352 | 0 |
| initial-observation open loop | 0/10 | 0/10 | 1.08153 | 0.11418 | 0 |

Raw recurrence and memoryless aligned prediction tie in the decisive success metric. This means temporal information can be useful in a conventional representation, while aligned latent recurrence adds neither success nor a lower endpoint error. The proposed model's 7/10 tie with both shuffled and reset history shows that its recurrent latent state is not the working mechanism. Physical shooting again spends thousands of candidate steps without improving reliability.

## Failed retries and audit

The two pre-result failures remain preserved. `EXP_G25` trained all models but failed before logging a case because the environment wrapper exposes `check_success()` rather than `_check_success()`. `EXP_G25_retry1` exposed a recording mismatch between the 16-D alignment coordinates used for ranking and the 64-D image latent required by the generic visual-residual calculation. `EXP_G25_retry2` records both spaces separately and is the formal result.

The independent audit passed with zero failures. It checked all 1,242 source branches, loaded all twenty equal-parameter checkpoints, recomputed 828 held-out temporal decisions, rebuilt 1,700 recurrent updates and normalized visual/physical/score feedback fields, verified 3,416 selected native proposal commits, checked all ninety rollout chains, and reconstructed all nine method aggregates. The exact decision-state feedback and the independent contiguous trajectory differed by at most 0.000641 m after checkpointed contact branches, below the declared 1 mm replay bound.

- `experiments/EXP_G25_retry2/temporal_sequence_manifest.json`
- `experiments/EXP_G25_retry2/fold_models/` and `model_selection.json`
- `experiments/EXP_G25_retry2/rollouts/`, `case_metrics.jsonl`, and `metrics.json`
- `experiments/EXP_G25_retry2/audit.json`
- `experiments/EXP_G25_retry2/audit_failed_feedback_tolerance.json`
- `scripts/experiments/run_exp_g25_temporal_policy_memory.py`
- `scripts/experiments/audit_exp_g25.py`

The post-EXP disk check left 850 GB free.

## Consequence for the system

G25 rejects a third candidate role for the current aligned action latent: lossy action decoding failed in G23, memoryless visual alignment was unnecessary in G24, and temporal latent memory now loses to raw history. The next experiment will not tune the GRU. G26 will test a distinct control function in which a learned transition coordinate represents epistemic support and uncertainty across bootstrapped outcome models, using uncertainty to avoid model-exploiting native proposals. It must beat a matched direct-action uncertainty ensemble to establish a latent contribution.
