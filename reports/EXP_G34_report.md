# EXP_G34 Report: Monotonic Execution-Phase Coordinate

## Conclusion

**NOT SUPPORTED: a learned monotonic action-phase coordinate did not improve autonomous switching or ordered success.** Under the fixed G33 physical-residual recovery controller, the monotonic coordinate completed 23/30 ordered `lift -> place` tasks. It lost to the matched direct binary F3 (24/30), state F3 (25/30), the deliberately shuffled-order coordinate (27/30), and the future-visible full-prompt controller (26/30). The shuffled ablation switched about 49 steps away from the retrospective boundary yet achieved the best endpoint result. Consequently, accurate single-task phase prediction and strict future-action gating are not the dominant mechanisms in this task.

## New representation and training evidence

G34 rebuilt a train-only phase dataset from 14 task-5 episodes at actual five-step policy issue points. The 562 aligned rows contain both camera views, policy state, a 16-action native-control history, episode identity, certified lift boundary, normalized progress, and completion label. Three matched 108,146-parameter models were trained for every leave-one-episode-out fold and once on all training episodes:

1. a 16-D monotonic phase coordinate trained with completion, progress, and temporal ranking objectives;
2. a direct binary completion model with the same encoder capacity;
3. a shuffled-order coordinate with within-episode progress/order targets permuted while retaining completion labels.

This produced 42 cross-fitted checkpoints and three final checkpoints. Cross-fitted rule selection chose threshold 1.1 with persistence 1 for the monotonic coordinate, 0.8 with persistence 1 for the binary model, and 1.0 with persistence 3 for the shuffled model. Their train-episode cross-fit mean boundary errors were 8.57, 0.36, and 40.71 policy steps, respectively; the shuffled model generated six premature switches.

## Prospective causal protocol

Three explicit-noise repeats over ten canonical held-out starts compared eight mechanisms in 240 prospective simulator rollouts. Every controller used the same native π0.5 proposal stream, three candidates, five-step replanning, two real 3 cm book displacements, exact simulator/controller checkpoint restoration, and G33's frozen raw-action physical-residual detector. All learned decisions consumed the observation and state reached by the preceding executed controller action.

The original G34 run included a method string named `oracle_f3`, but source inspection showed that this name was not handled by the G14 rollout implementation and therefore executed as a no-switch lift-only policy. Those 30 artifacts are preserved as invalid evidence and excluded from formal metrics. The same EXP was repaired by executing 30 new rollouts with an actual physical oracle that reads the current book height, switches once at the declared lift threshold, and records its online observations. The formal artifact merges only seven valid original methods with this repaired comparator.

| Method | Ordered success | Per-repeat | Mean absolute switch error | Rejected candidate steps |
|---|---:|---:|---:|---:|
| monotonic phase coordinate | 23/30 | 7, 8, 8 | 12.03 | 1,040 |
| direct binary F3 | 24/30 | 9, 8, 7 | 4.87 | 1,000 |
| **shuffled-order coordinate** | **27/30** | 9, 9, 9 | 49.17 | 900 |
| state F3 | 25/30 | 9, 8, 8 | 6.90 | 825 |
| outcome-latent F3 | 24/30 | 8, 8, 8 | 9.67 | **820** |
| fixed-time F3 | 23/30 | 8, 8, 7 | 13.20 | 915 |
| physical oracle F3 | 24/30 | 9, 7, 8 | **2.00** | 935 |
| future-visible full prompt | 26/30 | 10, 8, 8 | n/a, no explicit switch | 1,115 |

All methods completed the lift predicate in all 30 trials. In paired comparisons, the monotonic coordinate had one win/two losses against direct binary, no wins/two losses against state F3, no wins/four losses against shuffled order, and no wins/three losses against future-visible control. It therefore fails every predeclared superiority requirement and falls below the required 24/30 reference.

## Mechanism interpretation

The offline coordinate objective learned temporal order, but that property did not cause better control. The strongest method was the intentionally temporally corrupted coordinate, whose very early switches shortened committed execution (5,754 steps versus 6,378 for monotonic) and still left enough mixed language/control behavior to complete placement. Likewise, exposing the full future instruction throughout execution achieved 26/30 without an explicit switch. These are direct counterexamples to the original claim that future-action authority must be strictly hidden until a precise completion boundary on this benchmark.

The physical oracle's 2-step mean boundary error produced only 24/30, while much worse timing produced 27/30. Thus switch-error is not a reliable proxy for task success here. The result does not prove that action order is irrelevant in harder programs; it shows that the present two-stage task and policy are tolerant of early target information and cannot establish the intended current-action-protection mechanism. Further small latent heads on the same lift/place traces would not resolve this identifiability problem.

## Audit and artifacts

The independent audit passed. It rebuilt all 562 dataset rows, checked 42 fold models, recomputed 1,686 held-out predictions, 1,236 online phase decisions, 500 physical-oracle decisions, 11,647 physical recovery trials and exact restores, 30,411 proposal-noise seeds, 10,137 native commits, 480 physical interventions, all 240 prospective chains, every aggregate, paired comparison, and the negative support conclusion.

- `experiments/EXP_G34/phase_dataset.npz`, `dataset_manifest.json`, `model_selection.json`, fold checkpoints, final checkpoints, and original rollouts
- `experiments/EXP_G34_oracle_repair/repair_case_metrics.jsonl`, `formal_case_metrics.jsonl`, `metrics.json`, repaired oracle rollouts, and `audit.json`
- `experiments/EXP_G34_smoke/` and the invalid original oracle artifacts are preserved but are not counted as formal evidence
- `scripts/experiments/run_exp_g34_monotonic_phase_coordinate.py`
- `scripts/experiments/repair_exp_g34_physical_oracle.py`
- `scripts/experiments/audit_exp_g34.py`

Reproduction commands are recorded in each script header. The formal audit command was:

```bash
PYTHONPATH=src:scripts/experiments:/home/jinjaguo/LIBERO \
NUMBA_CACHE_DIR=/tmp/pglt_numba_cache_g34_audit \
/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/audit_exp_g34.py \
  --formal experiments/EXP_G34_oracle_repair
```

The post-EXP disk check left 847 GB free.

## Consequence for the system

The best evidence-backed F2 remains raw-action physical residual recovery, while no learned action latent has yet made a meaningful control contribution. G34 additionally shows that the existing two-stage program is too forgiving to identify current-action protection from success alone. G35 must therefore change the task construction: create an actual three-stage `grasp -> lift -> place` program with two autonomous boundaries and multi-action training data, so premature future authority has a physically measurable opportunity to break the unfinished grasp or lift.
