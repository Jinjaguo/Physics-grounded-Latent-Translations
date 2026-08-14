# PGLT wave-16 amended public-data H1/H2 external replication

## Scope

This is the post-audit amendment recorded in `prompts/dynamics_4.md`. It evaluated **H1 and H2 only** using four strict non-overlapping H16 windows per public CALVIN task segment. **H4 and H8 were not run.** Representation, semantic predictor, F1, and F2 remained frozen; DEL was not run.

## Data

- Source: `VyoJ/calvin-ABCD-D-subsets`.
- Processed shards: `subset_training_023`, then `subset_training_000`; stopped at the preregistered gate.
- Selected: 60 direct annotation-consistent segments, exactly 10/task.
- Segment length: 64–65 frames; exactly the first 64 frames form four stride-16 windows, with no padding.
- Rollout starts: H1 = 120; H2 = 60.

## Primary paired trajectory endpoint

- F1 mean normalized H1/H2 trajectory AUC: **0.728379**.
- F2 mean normalized H1/H2 trajectory AUC: **0.632797**.
- Delta AUC (F2-F1): **-0.095582**, 95% CI **[-0.113417, -0.079462]**.
- Preregistered external replication gate: **PASS**.

## Horizon and mechanism metrics

| metric | F1 H1 | F2 H1 | F1 H2 | F2 H2 |
|---|---:|---:|---:|---:|
| execution MSE | 0.749443 | 0.670658 | 0.776540 | 0.655076 |
| decoded continuous MSE | 0.028221 | 0.025877 | 0.038212 | 0.034635 |
| execution kNN radius | 2.268301 | 2.141792 | 1.901186 | 1.703278 |

Mean refinement correction-target cosine: **0.415405**; positive fraction: **0.944**.

## Claim decision

C3c-local: **STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION**. C3c-long remains **NOT TESTED** because this amended experiment deliberately ran H1/H2 only. H4/H8 results must not be inferred from this replication.
