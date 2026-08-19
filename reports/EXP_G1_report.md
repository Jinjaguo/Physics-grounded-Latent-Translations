# EXP_G1 Report: Recoverable Action-Conditioned LIBERO Transition Benchmark

## Conclusion

**SUPPORTED: the repository can now produce a reusable, exactly restorable, action-conditioned causal transition benchmark through the real LIBERO simulator/controller interface.** This conclusion is limited to the transition primitive. EXP_G1 does not establish that F2 improves control, that F3 can switch autonomously, or that the final `lift -> place` system works.

## Scientific hypothesis and necessity

The hypothesis was that the existing full-state snapshot implementation could restore a physically meaningful LIBERO control boundary closely enough to compare multiple interventions from the same starting state, and that different executable control proposals would produce measurably different realized next states while the learned action representation could be re-encoded after each executed step.

This was the required first experiment because all later F2 comparisons depend on a valid `checkpoint + proposal -> executed action -> realized feedback -> next latent` primitive. Old Wave19 certification showed snapshot replay in an earlier protocol, but it was not renamed or counted as G-series evidence. EXP_G1 selected new development branches, executed a new multi-proposal protocol, and wrote a new step-level causal dataset.

## Exact data and split

- Source split definition: `results/dynamics/nineteenth_wave/2026-08-14_dynamics_7/wave19_dataset_split_manifest.json`.
- Selection split: development only; no test episode was opened.
- States: one certified `branch_025` checkpoint from each of the ten official LIBERO-10 tasks, for ten distinct source episodes.
- Checkpoint payload: MuJoCo `mjSTATE_INTEGRATION` plus robosuite environment counters, controller fields, robot buffers, gripper state, and observable state through `pglt.libero.snapshot.LiberoSnapshot`.
- Active action language: the source episode instruction saved with every intervention record.
- Learned representation: frozen Wave20 LIBERO correct-language EMA model, seed 202820, with its frozen train-only continuous-action normalization.

## Intervention protocol

Every selected checkpoint was restored before every rollout. The experiment compared four genuinely different executable proposal families over 16 controller steps:

- `source`: the recorded successful π0.5 continuation prefix.
- `damped`: half-amplitude continuous motion with the same gripper commands.
- `reverse`: sign-reversed continuous motion with the same gripper commands.
- `hold`: zero continuous motion with the same gripper commands.

The source proposal was independently restored and executed three times per checkpoint. Thus each checkpoint produced six rollouts: three source repeats and one rollout for each of the other three proposals. Every environment step used a private action copy through `safe_env_step`; the next decision state was obtained from the simulator after that executed step, never from an offline latent trajectory.

Each compressed rollout archive contains requested and executed controls, complete integration state, qpos/qvel, end-effector pose, gripper state, body positions, object observation, agent-view and wrist RGB observations, reward, done/success state, and the frozen representation latent re-encoded from the most recent 16 actually executed controls. The action-only encoder limitation is discussed below.

## Quantitative results

| Metric | Result |
|---|---:|
| Tasks / checkpoints | 10 / 10 |
| Intervention rollouts | 60 |
| Actually executed controller steps | 960 |
| Same-action repeat comparisons | 20 |
| Maximum same-action integration-state discrepancy | 0.0 |
| Maximum same-action latent discrepancy | 0.0 |
| Median distinct-action integration endpoint L2 | 4.738853 |
| Minimum distinct-action integration endpoint L2 | 0.270518 |
| Median distinct-action end-effector endpoint L2 | 0.075445 m |
| Median distinct-action object-observation endpoint L2 | 0.255963 |
| Median distinct-action latent endpoint L2 | 5.111248 |
| Distinct pairs above repeat tolerance | 100% |
| Finite-value audit | PASS |

The complete full-state restore therefore reproduced all repeated source rollouts exactly in this protocol. Every pair of distinct proposals produced an integration-state endpoint separation larger than the observed repeat tolerance. The benchmark is sufficiently sensitive to distinguish interventions rather than merely replaying a fixed next state.

## Independent integrity audit

`scripts/experiments/audit_exp_g1.py` independently reopened every NPZ, checked all required causal fields, verified that executed controls equal the logged requests, checked action/feedback time bases and finite values, verified development-only membership, and recomputed the principal metrics. It checked 60 unique artifacts and reported zero failures. The recomputed values exactly matched `metrics.json`.

The completion-time disk check found approximately 850 GB free, above the required 200 GB floor. EXP_G1 occupies approximately 8.7 MB.

## Machine-verifiable artifacts

- Runner: `scripts/experiments/run_exp_g1_transition_benchmark.py`
- Independent auditor: `scripts/experiments/audit_exp_g1.py`
- Exact invocation, git revision, dirty-worktree record, and source paths: `experiments/EXP_G1/run_metadata.json`
- Runtime versions and environment variables: `experiments/EXP_G1/environment.json`
- Dataset definition and intervention-to-artifact links: `experiments/EXP_G1/dataset_manifest.json`
- Flat causal log: `experiments/EXP_G1/interventions.jsonl`
- Raw/minimally processed rollouts: `experiments/EXP_G1/transitions/*.npz`
- Aggregate metrics: `experiments/EXP_G1/metrics.json`
- Per-checkpoint repeated-replay results: `experiments/EXP_G1/reproducibility_details.json`
- Per-checkpoint intervention separation: `experiments/EXP_G1/causal_separation_details.json`
- Independent audit result: `experiments/EXP_G1/audit.json`

## Reproduction commands

```bash
PYTHONPATH=src:/home/jinjaguo/LIBERO \
MUJOCO_GL=egl \
NUMBA_CACHE_DIR=/tmp/pglt_numba_cache \
MPLCONFIGDIR=/tmp/pglt_matplotlib \
/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/run_exp_g1_transition_benchmark.py \
  --output experiments/EXP_G1 \
  --branches-per-task 1 \
  --horizon 16 \
  --same-action-repeats 3 \
  --device cpu

/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/audit_exp_g1.py \
  --experiment experiments/EXP_G1 \
  --tolerance 1e-12
```

The runner intentionally refuses to overwrite an existing output directory. Reproduction should therefore use a new destination when preserving the released EXP_G1 evidence.

## What was learned and what was not

Direct evidence supports exact checkpoint restoration, real executable proposal branching, realized observation/state capture, and deterministic re-encoding in this environment. This removes the simulator/checkpoint gate that blocked earlier work.

EXP_G1 does not show that any learned planner is better than a baseline. The proposal set was deliberately diagnostic, and `source` uses recorded future controls. No task-completion superiority claim follows from endpoint separation. The representation is also action-only: its post-step latent is re-encoded from the rolling window of executed control commands. Although this latent changes after real interventions and is paired with complete realized feedback, it is not yet the desired `E(o_t, q_t, g_t)` observation-conditioned execution state. Later experiments must either add physical-state conditioning or show explicitly how the action latent and physical feedback jointly affect selection.

No F3 model, current-action protection mechanism, retargeting, autonomous switch, two-action sequence, or recovery controller was evaluated. The final acceptance goal remains unmet.

## Decision for EXP_G2

The dominant bottleneck is no longer checkpoint restoration. It is whether F2-like proposal generation and refinement helps when its decoded controls are actually executed and replanned from realized state. EXP_G2 will therefore compare the independently trained LIBERO F1 and F2 checkpoints against direct action/state baselines under oracle completion boundaries, using receding-horizon simulator feedback from matched checkpoints. The saved realized state—not an offline teacher latent—must determine subsequent proposal evaluation and selection.
