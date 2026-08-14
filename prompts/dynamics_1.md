# PGLT Thirteenth-Wave Codex Prompt — Frozen Latent Dynamics: Equal-Information MLP vs Variational Dynamics

## Scope

The representation phase is complete for the current paper.

The twelfth-wave frozen prospective replication passed:

- representation-ready = TRUE
- representation frozen = TRUE
- latent dynamics authorized = TRUE

Historical strict Gate A remains FAIL and must remain preserved in provenance. Representation readiness was established later by the separately preregistered prospective R-Gate.

This wave is the **first scientific latent-dynamics experiment**.

Do not modify, retrain, fine-tune, or reselect the representation. Do not alter EMA. Do not search representation hyperparameters. Do not use future target actions in any primary dynamics model.

Before implementing anything, inspect:

- `twelfth_wave_results.md`
- `twelfth_wave_next_experiment.md`
- `representation_readiness_decision.json`
- `r_gate_decision.json`
- the twelfth-wave final integrity artifact
- the frozen wave-11 EMA checkpoint manifest
- the fifth-wave causal-control alignment note
- the corrected local LaWM adapter
- all frozen representation configs
- exact CALVIN episode/action metadata
- current latent encoder/decoder APIs
- all existing dynamics design notes and tests

Do not guess paths, checkpoint identifiers, tensor layouts, action timing, latent dimension, episode splits, timestep, solver coefficients, or metric implementations. Artifacts and source are authoritative.

## 1. Scientific question

The frozen representation has already demonstrated language addressability, cross-episode semantic robustness, motor reconstruction fidelity, seed robustness under EMA, and prospective replication.

The new question is:

**Can the frozen continuous action latent serve as a useful dynamical coordinate, and does a variational transition rule improve long-horizon latent rollouts relative to generic neural prediction under the same information set?**

The scientific comparison must isolate the effect of variational structure. Do not let future-control leakage or overlapping latent windows make the task artificially easy.

## 2. Freeze the representation permanently

Use the exact frozen wave-11 EMA representation.

Freeze:

- EMA checkpoint identity
- epoch 40
- EMA decay
- architecture
- latent dimension 32
- 16/16 semantic/execution factorization
- action-only encoder
- H=16 action window
- decoder
- shared-trunk CLIP isolation
- text tower
- text projection
- normalization
- all representation parameters

During this wave:

```text
representation optimizer steps = 0
representation backward calls = 0
EMA updates = 0
```

Encode latents under `torch.no_grad()` and save hashes of all frozen checkpoints used.

## 3. Primary latent timing must use non-overlapping windows

Define

```text
q_k = E_A(a[t_k : t_k+H])
H = 16
```

with

```text
t_{k+1} = t_k + H
```

Therefore the primary dynamics stride is exactly:

```text
stride = H = 16
```

unless the frozen representation artifact proves a different H.

Do NOT use stride-one sequences in the primary benchmark. With stride one, adjacent coordinates share 15/16 actions and one-step prediction can become artificially easy.

Stride-one may be retained only as an explicitly labeled overlap diagnostic after the primary results are frozen.

Verify the exact CALVIN control frequency from source metadata and report the physical time represented by one latent step.

## 4. Prediction issue time and causal information

For a latent q_k encoded from action frames `[t_k, t_k+H-1]`, define the prediction issue frame:

```text
tau_k = t_k + H
```

At `tau_k`:

- q_k is available
- q_{k-1} is available
- known language/task context is available
- raw logged actions with frame index < tau_k are causal history
- actions from the target future window `[tau_k, tau_k+H-1]` are unavailable

Primary models must never consume target-window future actions.

Add runtime causal masks that reject any control index >= prediction issue frame.

## 5. Dynamics dataset audit before training

Build a dynamics-specific dataset audit from the frozen six-task CALVIN data.

Representation is frozen, so dynamics may use its own episode split.

Use:

- official `task_D_D` training episodes for dynamics train/development
- official `task_D_D` validation episodes as final held-out dynamics evaluation

State transparently that official validation episodes may have been used previously for representation evaluation, but they have never been used to fit dynamics models.

Before training, enumerate all non-overlapping latent sequences satisfying the frozen six-task criterion.

For every sequence record:

- episode row
- annotation/task ID
- language string
- global frame range
- q-window frame indices
- number of non-overlapping latent steps
- whether all windows lie inside one annotation
- whether a task boundary occurs
- physical duration

Primary training/evaluation must use windows fully contained in one annotation/task segment so context is well-defined.

Create a separate task-boundary diagnostic only if enough valid adjacent segments exist. Do not silently mix within-task and boundary transitions.

## 6. Dynamics train/development split

Using official training episodes only, create a deterministic episode-level dynamics train/development split before model fitting.

Preferred:

- fixed deterministic episode permutation
- approximately 75–80% dynamics train
- remaining episodes dynamics development

Never split transitions from the same episode across train and development.

Write `dynamics_split_preregistration.json` before training, including:

- exact train episode rows
- exact development episode rows
- deterministic rule and RNG/meta-seed if used
- sequence/transition counts
- task coverage
- horizon coverage
- reserved official validation rows
- explicit statement that no dynamics-test metrics have been read

Do not tune on official validation results.

## 7. Frozen latent serialization

Precompute frozen latents once.

For every valid window save:

- full 32-D q_k
- optional 16-D z_sem and z_exec slices
- source action-window indices
- task/language context identifier
- episode identifier
- issue frame
- causal-history action indices
- target latent index
- checksum/provenance

Primary dynamics models operate on full 32-D q.

Do not train separate subspace dynamics in this wave.

## 8. Context representation

Use one frozen context representation for all learned dynamics models.

Preferred context is the frozen language embedding or frozen projected semantic text embedding already used by the representation. Inspect exact saved code and preserve it.

Do not retrain the text tower.

For within-task sequences, context stays fixed through the rollout.

Do not give different context information to different primary models.

## 9. Primary comparison Block A — autonomous equal-information models

All Block-A models receive exactly:

```text
(q_{k-1}, q_k, c_k)
```

and no raw control packet.

Run at minimum:

### A0. Copy baseline

```text
q_hat_{k+1} = q_k
```

### A1. Constant-velocity latent baseline

```text
q_hat_{k+1} = q_k + (q_k - q_{k-1})
```

### A2. MLP transition baseline

```text
q_hat_{k+1} = F_phi(q_{k-1}, q_k, c_k)
```

Use modest capacity matched as closely as practical to the variational model. Do not provide future controls.

### A3. Corrected unforced DEL transition

Learn a discrete Lagrangian `L_d_theta(q_k, q_{k+1}; c_k)` and predict q_{k+1} through the corrected implicit DEL equation:

```text
D2 L_d(q_{k-1}, q_k; c_k) + D1 L_d(q_k, q_{k+1}; c_k) = 0
```

Use the corrected sign/time-scaled solver principles from the local LaWM adapter. Do not reuse the upstream unstable update.

All Block-A learned models must use exactly the same training/development/test transitions.

## 10. Discrete Lagrangian parameterization

Use a conservative 32-D parameterization:

```text
L_d(q_k,q_{k+1};c)
= h * [0.5 * v^T M_theta(q_bar,c) v - V_theta(q_bar,c)]

v = (q_{k+1}-q_k)/h
q_bar = (q_k+q_{k+1})/2
```

For this first dynamics wave use a diagonal positive mass:

```text
M_theta = diag(softplus(m_theta) + epsilon)
```

Do not use a full 32×32 mass matrix in the first experiment.

Potential `V_theta` is scalar. Context may condition mass and potential.

Report exact parameter counts.

Do not call this learned mass a physical robot inertia matrix. It is a metric-like kinetic term in the learned latent coordinate system.

## 11. DEL solver requirements

The solver must:

- be differentiable
- remain finite under multi-step unrolling
- use corrected sign/time scaling
- expose residual norm per iteration
- expose convergence/failure status
- never hide instability with `nan_to_num`
- never silently clamp target latents
- never substitute ground-truth future q during rollout

Before real CALVIN latent training, pass:

1. existing official LaWM toy finite regression
2. synthetic constant-mass free-particle test
3. synthetic quadratic-potential test
4. 32-D random finite-gradient smoke test
5. multi-step finite-rollout test

## 12. Training objectives

MLP objective:

```text
L_MLP = ||q_hat_{k+1} - q_{k+1}||^2
```

DEL primary objective:

```text
L_pred = ||q_hat_{k+1} - q_{k+1}||^2
```

A small fixed DEL-residual term may be included only if required:

```text
L = L_pred + lambda_DEL * ||R_DEL||^2
```

If used, preregister exactly one lambda before training. Do not sweep a large grid.

Record all objective terms.

## 13. Matched refinement-control baseline

DEL uses iterative/implicit refinement.

If technically feasible without creating a second large project, add a matched generic refinement baseline:

- initialize from the MLP prediction
- use the same number of refinement steps
- optimize a learned generic transition energy/objective without DEL structure

Purpose:

**Test whether DEL helps beyond simply spending more inference-time optimization steps.**

If a fair matched baseline is not feasible, document this limitation and report solver compute explicitly.

## 14. Comparison Block B — causal-history conditioned models

The current latent definition does not provide a clean independently observed physical generalized force.

Construct a `ControlPacket` containing only already executed actions:

- values
- command_frame_indices
- prediction_issue_frame
- availability_source
- available_before_prediction

Reject any packet containing command indices >= prediction issue frame.

Run:

### B1. History-conditioned MLP

```text
q_hat_{k+1} = F_phi(q_{k-1}, q_k, c_k, u_k^-)
```

### B2. Causal-history forced DEL

```text
D2 L_d(q_{k-1},q_k;c_k)
+ D1 L_d(q_k,q_{k+1};c_k)
+ Q_psi(q_{k-1},q_k,c_k,u_k^-)
= 0
```

B1 and B2 must receive the exact same causal-history packet.

Do not compare B2 against A2 and attribute gains to variational structure when their information sets differ.

Primary controlled comparison is:

```text
B1 history-conditioned MLP
vs
B2 causal-history forced DEL
```

State clearly that this raw history may duplicate information already represented in q_k. This is a history-conditioned latent forcing model, not proof of independently observed physical force.

## 15. Strictly separate oracle future-action diagnostic

Optionally run an oracle diagnostic using logged future target-window actions.

It must be named:

```text
ORACLE_FUTURE_ACTION_DIAGNOSTIC
```

It must be excluded from:

- primary rankings
- autonomous claims
- controlled-model claims
- deployment comparisons

Any table/figure must label it as a future-action leakage upper bound.

Do not let oracle results determine model selection.

## 16. Dynamics development metrics

Use dynamics development episodes only for model selection.

### One-step latent prediction

Report:

- full-latent MSE
- cosine similarity/error
- z_sem MSE
- z_exec MSE

Subspace metrics are evaluation only; models predict full q.

### Multi-step autonomous rollout

Evaluate horizons:

```text
1, 2, 4, 8
```

and horizon 16 only if enough sequences support it.

Report sample count and physical duration at each horizon. Never pad missing trajectories.

### Decoded action fidelity

Decode predicted latents through the frozen decoder.

Report:

- continuous action MSE
- gripper accuracy
- all seven per-action-dimension errors

Do not retrain the decoder.

### Semantic retention

Where task labels exist, report:

- predicted-latent → text retrieval
- correct task assignment
- semantic cosine to frozen task prototype

### Off-manifold behavior

Report:

- nearest dynamics-training-latent distance
- kNN radius/density
- ratio relative to ground-truth held-out latent distribution
- fraction beyond a preregistered training-distribution quantile

Choose the quantile on train/development only.

### Numerical stability

For DEL models report:

- DEL residual norm
- solver iterations
- convergence rate
- nonfinite rate
- learned energy drift/change as diagnostic only

Do not interpret learned energy as physical energy conservation.

## 17. Primary long-horizon criterion

Do not select a winner from one-step MSE alone.

Predefine the primary ranking metric as:

**Area under rollout-error-vs-horizon curve over the preregistered horizons, using full-latent MSE normalized by ground-truth latent variance estimated from dynamics training data.**

Also report raw MSE at every horizon.

A model provides meaningful long-horizon improvement only if:

- one-step performance is competitive
- error grows more slowly with horizon
- decoded actions remain more accurate
- off-manifold drift is reduced

Do not claim useful dynamics from low DEL residual alone.

## 18. Dynamics model selection

Use only dynamics development episodes.

Freeze before official validation evaluation:

- model checkpoints
- solver iterations
- DEL coefficients
- refinement settings
- off-manifold threshold
- rollout horizons
- metric code

Write immutable `dynamics_confirmation_manifest.json` containing hashes and settings.

Official validation dynamics evaluation is one-shot.

Do not tune after test metrics are read.

## 19. Held-out dynamics evaluation

After manifest freeze, evaluate on official CALVIN validation episodes.

Report separately:

### Block A
- copy
- constant velocity
- MLP
- unforced DEL
- matched refinement baseline if available

### Block B
- history-conditioned MLP
- causal-history forced DEL

### Oracle
- future-action teacher-forced diagnostic, if run

Do not merge Block A and Block B into one leaderboard without marking their different information sets.

## 20. Task-boundary / multi-stage diagnostic

After primary within-task results are frozen, audit adjacent CALVIN annotations in the same episode.

A boundary example is eligible only if:

- both tasks are among the frozen six
- source/target latent windows are non-overlapping
- exact gap is known
- no target future action is leaked

Freeze the eligibility rule and list before evaluation.

Evaluate frozen dynamics models without retraining.

If too few eligible boundaries exist, report insufficient support rather than loosening criteria.

## 21. Interpretation rules

### Result A — DEL improves long-horizon rollout
If unforced DEL has similar one-step error but lower multi-step AUC, lower decoded-action drift, and lower off-manifold drift than MLP under identical autonomous inputs, support a variational inductive-bias claim.

### Result B — forced DEL improves over history-MLP
If causal-history forced DEL improves over history-conditioned MLP under identical causal inputs, support a structured controlled-transition claim.

Do not call causal history a true physical force unless independently justified.

### Result C — DEL reduces residual only
If DEL residual improves but rollout does not, do not claim useful dynamics improvement.

### Result D — MLP wins
Report it. Do not alter the frozen representation post hoc.

## 22. Representation remains frozen regardless of dynamics outcome

The twelfth-wave representation-readiness decision is final for this paper.

Do not reopen representation search because a dynamics model underperforms.

If variational dynamics fails, diagnose the dynamics formulation.

## 23. Tests

Add/preserve tests for:

- frozen representation hashes
- zero representation gradients
- non-overlapping primary windows
- exact stride=H timing
- causal control mask
- future target-action rejection
- episode split isolation
- no official validation metrics before manifest
- MLP/DEL information-set equality
- B1/B2 control-packet identity
- DEL finite gradients
- synthetic solver behavior
- multi-step rollout without teacher forcing
- decoder frozen during predicted-latent evaluation
- oracle exclusion from primary aggregate
- corrected LaWM toy regression

Do not modify third-party CALVIN or LaWM repositories.
Do not overwrite prior-wave artifacts.
Create a timestamped thirteenth-wave dynamics directory.

## 24. Required deliverables

Produce:

- `thirteenth_wave_results.md`
- `thirteenth_wave_next_experiment.md`
- dynamics dataset audit
- dynamics split preregistration
- frozen latent serialization manifest
- representation checkpoint hash audit
- Block-A model specs
- Block-B model specs
- parameter-count table
- solver numerical-validation report
- one-step development metrics
- rollout development metrics
- decoded-action development metrics
- off-manifold development metrics
- frozen dynamics confirmation manifest
- held-out dynamics evaluation
- rollout-error-vs-horizon tables
- semantic-retention tables
- DEL residual/stability report
- causal information-set audit
- task-boundary diagnostic if eligible
- oracle diagnostic clearly separated if run
- exact commands executed
- environment/provenance report
- files-changed report
- updated tests
- updated `RESEARCH_LOG.md`
- updated `NEXT_EXPERIMENT.md`

The final report must explicitly answer:

1. How many non-overlapping frozen-latent sequences were available for train/development/test?
2. What physical duration does one latent step represent?
3. Were any primary windows overlapping?
4. Did any primary model access future target actions?
5. Under identical autonomous information, does unforced DEL outperform MLP at one step?
6. Does unforced DEL outperform MLP over long rollout horizons?
7. Does DEL reduce off-manifold drift?
8. Do predicted latents decode to more accurate future action chunks?
9. Does semantic task information remain stable during rollout?
10. Under identical causal-history inputs, does forced DEL outperform history-conditioned MLP?
11. Did the corrected DEL solver remain finite and convergent?
12. Did any apparent forced-model advantage depend on future-action leakage?
13. What happened at task boundaries?
14. Does the evidence support treating the frozen representation as a useful dynamical coordinate?
15. Does the evidence support a variational inductive bias in that coordinate system?
16. What is the single next dynamics experiment justified by the results?

Do not write the desired conclusion in advance.

Scientific correctness, causal information-set equality, and numerical auditability are the priority.
