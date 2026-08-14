# PGLT Eighteenth-Wave Codex Prompt
# Prospective Closed-Loop Embodied Validation and Causal Refinement Intervention

## 0. Mission

This wave is the decisive embodied validation for the current paper.

The paper story is now:

> **Language grounds meaningful and executable action coordinates. A free predictor advances the latent trajectory, while iterative refinement suppresses accumulated drift and keeps long-horizon evolution near empirically executable regions.**

Wave 18 must test whether the offline latent-space advantages already established in waves 15–17 translate into **real closed-loop robot behavior inside the CALVIN simulator**.

This is not another representation wave.
This is not another DEL wave.
This is not an architecture-search wave.

The representation, semantic predictor, F1 execution MLP, and F2 matched refinement must remain frozen.

The key scientific question is:

> **When predicted latents are decoded into robot commands and physically executed, does the same frozen refinement operator causally improve continuation behavior, reduce rollout failure, and recover from proposal perturbations better than an unconstrained predictor?**

The experiment must also distinguish:

```text
benefit from the learned refinement direction
vs
benefit from merely spending four extra iterative computation steps
```

This distinction is essential for the ICLR paper.

---

# 1. Frozen Scientific State

Preserve the following claim state exactly:

```text
C1_language_addressable_coordinates = SUPPORTED
C2_continuously_decodable_coordinates = SUPPORTED

C3a_full_latent_DEL = REJECTED
C3b_execution_DEL = REJECTED

C3c_local_refinement =
STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION

C3c_long_refinement = SUPPORTED

C3d_empirical_manifold_stabilization = SUPPORTED_WITH_NONMONOTONIC_LOCAL_NUANCE

context_dependency = ROBUST_TO_BOUNDARIES
```

Historical negative findings must remain visible.

Do not reopen DEL.

Do not retrain the representation.
Do not retrain F1.
Do not retrain F2.
Do not modify the semantic predictor.
Do not change F2 iteration count.
Do not tune F2 using wave-18 results.

Required:

```text
representation optimizer steps = 0
semantic predictor optimizer steps = 0
F1 optimizer steps = 0
F2 optimizer steps = 0
EMA updates = 0
```

Before any embodied rollout, record SHA256 hashes for:

- representation checkpoint;
- semantic predictor checkpoint;
- F1 checkpoint;
- F2 checkpoint;
- frozen decoder;
- normalization statistics.

Write:

`wave18_frozen_model_manifest.json`

---

# 2. Important Scope Limitation

The action latent is action-only.

The representation maps:

```text
16-frame executed action chunk -> latent
```

It does not map the current RGB/state observation directly to a next action latent.

Therefore wave 18 must NOT claim:

```text
full autonomous policy from task start
```

The correct embodied evaluation unit is:

> **closed-loop continuation from a causally observed action-history branch point.**

At every branch point, both F1 and F2 receive exactly the same causal action-latent history and context.

Then the models recursively predict future latent chunks, decode them to 7-DoF CALVIN actions, and execute those actions in the simulator.

This directly tests the current dynamics claim.

Suggested terminology:

```text
closed-loop latent continuation
```

Do not call it end-to-end policy evaluation.

---

# 3. Phase 0 — Exact Simulator Reconstruction Gate

Before any model comparison, prove that the CALVIN environment can be reconstructed/replayed sufficiently accurately from the chosen source trajectories.

This gate is mandatory.

## 3.1 Audit CALVIN reset/state APIs

Inspect the exact installed CALVIN source.

Identify:

- environment reset interface;
- robot initial-state interface;
- scene/object state interface;
- simulator internal state required for deterministic continuation;
- controller internal state;
- gripper state;
- timing/counters;
- random seeds;
- action preprocessing;
- success predicate evaluation.

Do not assume `robot_obs` + `scene_obs` is sufficient.

Document the exact source files and functions.

Produce:

`calvin_closed_loop_state_audit.md`

## 3.2 Preferred reconstruction strategies

Use the strongest available method in this order:

### A. Exact simulator snapshot/restore

If CALVIN/MuJoCo exposes sufficient simulator state, serialize the complete state required for deterministic continuation.

### B. Deterministic replay from a known reset

If exact snapshot is unavailable, reset to the source episode initial state and replay the original recorded action stream to the branch frame.

### C. Official CALVIN episode initialization

If the dataset/task infrastructure exposes exact episode-start initial conditions, use those plus deterministic replay.

Do not invent an approximate reset silently.

## 3.3 Zero-intervention twin replay

For every candidate branch point used in the embodied study:

1. create two simulator twins from the same reconstructed state;
2. apply the exact same source continuation actions;
3. compare their state trajectories.

Measure:

- joint positions;
- joint velocities if available;
- TCP pose;
- gripper state;
- object poses;
- scene state;
- contact-related state if exposed;
- final task predicate.

Required reconstruction gate:

```text
100% terminal task-predicate agreement
no nonfinite values
median state twin error effectively zero
P95 below a preregistered numerical tolerance
```

The exact tolerance must be determined from deterministic source replay, not model performance.

If deterministic branch replay is impossible, STOP.

Do not continue to embodied model comparison.

Instead write:

`wave18_reconstruction_gate_failure.md`

The paper may still use wave17 offline results, but wave18 embodied claims are unauthorized.

---

# 4. Source Data for Closed-Loop Branches

Use only trajectories whose provenance and task predicates are known.

Preferred source order:

1. official CALVIN held-out validation/test episodes not used for training F1/F2;
2. frozen public VyoJ CALVIN trajectories whose simulator state can be reconstructed exactly;
3. additional official CALVIN environment-generated trajectories collected prospectively, only if needed.

Do not choose branches based on F1/F2 performance.
Do not cherry-pick visually easy continuations.

Write the branch-selection rules before model inference.

---

# 5. Six Primary Tasks

Use the same six canonical tasks where possible:

```text
lift_blue_block_slider
lift_red_block_table
place_in_slider
push_pink_block_right
turn_off_lightbulb
turn_on_lightbulb
```

If exact embodied reconstruction is impossible for one task, report that explicitly.
Do not silently replace tasks after seeing model performance.

Target:

```text
>= 30 independent source episodes per task
```

Preferred:

```text
50 per task
```

Minimum confirmatory target:

```text
>= 180 independent source episodes total
```

with no task contributing more than 35% of the total.

Bootstrap/statistical unit must be the source episode, not individual branch points.

---

# 6. Branch-Point Design

Each source episode may contribute multiple branch points, but these are nested within the same episode and must not be treated as statistically independent.

Choose branch points prospectively based on trajectory time only.

Primary branch fractions:

```text
25%
50%
75%
```

of the source task continuation, subject to having enough future horizon.

If a trajectory is too short for all three:

- use only eligible preregistered fractions;
- do not move the branch after seeing prediction quality.

Also classify branch points by physical phase if source metadata permits:

```text
pre-contact
contact/manipulation
post-contact/transport
```

This phase analysis is secondary.

---

# 7. Causal Warm-Start

At branch point k, provide both models with the same causal latent history:

```text
z_{k-1}
z_k
current semantic/context input c_k
```

These latent states must be encoded only from actions already executed before the prediction issue time.

No future source actions may enter the predictor.
No future source robot state may enter the predictor.
No future task label may be injected unless it is available causally at the branch point.

If the branch crosses a later language boundary, Protocol-A rules from wave17 apply:

```text
hold the causally known start context
do not inject future annotation changes
```

A separate exogenous-context diagnostic is allowed, but it is secondary and must be clearly labeled.

---

# 8. Closed-Loop Execution Protocol

At each latent step j:

1. F1 receives the current causal latent history and context.
2. F1 predicts the next execution latent.
3. F2, when used, starts from that exact F1 prediction and applies the exact frozen four-step matched refinement.
4. Combine the predicted execution latent with the exact frozen semantic prediction.
5. Decode the full latent to a 16-frame CALVIN action chunk.
6. Apply the exact action postprocessing used by the frozen representation/decoder.
7. Execute the 16 actions sequentially in the simulator.
8. Record the actual simulator state after execution.
9. Advance to the next latent step using only causally available model-side information.

Important:

The model is action-history based.

Do not secretly encode simulator state into the next latent.

The next model latent history is based on the model's own previously generated/decoded action chunks, not ground-truth future actions.

This preserves the meaning of autonomous latent continuation.

---

# 9. Primary Embodied Models

Primary comparison:

```text
E0 = source/expert continuation upper reference
E1 = F1 free predictor
E2 = F2 matched refinement
```

The source/expert continuation is not a learned baseline.

It provides:

- task-success ceiling;
- trajectory reference;
- replay integrity reference.

Do not include historical DEL in the primary embodied comparison.
DEL remains a historical negative baseline only.

---

# 10. Mandatory Causal Intervention Controls

This section is essential.

The paper currently claims that the learned refinement direction stabilizes evolution.

Wave 18 must rule out the explanation:

> "F2 wins only because it gets four extra iterative computation steps."

Construct matched controls using the exact same F1 initial prediction.

## E3 — Norm-matched random correction

At every refinement step, replace the learned correction direction with a random vector.

Requirements:

- same correction norm as F2 at that step;
- deterministic preregistered random seed schedule;
- same number of four correction steps;
- same compute budget where practical;
- no future information.

Call:

```text
RANDOM_NORM_MATCHED_REFINEMENT
```

## E4 — Negative refinement

Apply the opposite of the learned F2 correction:

```text
delta_neg = - delta_F2
```

using the same four-step schedule.

If iterative dependence makes exact negation ambiguous, define before evaluation:

```text
negative correction at each step =
negative of the frozen F2 update evaluated at the current negative-control state
```

Document exact implementation.

Call:

```text
NEGATIVE_REFINEMENT
```

## E5 — Direction-shuffled learned correction

Build a library of learned correction vectors from development/training data only.

At evaluation, apply a norm-matched correction vector from another unrelated source state/task.

No evaluation-set correction may be reused across samples.

Call:

```text
SHUFFLED_LEARNED_DIRECTION
```

This control asks whether:

```text
learned local direction matters
```

rather than only correction magnitude/statistics.

## Primary mechanistic ordering hypothesis

Expected only if supported:

```text
F2 learned refinement
>
F1
>
random / shuffled / negative
```

Do not hard-code the conclusion.

---

# 11. Latent-Perturbation Recovery Experiment

The strongest current mechanism claim is stabilization near executable regions.

Test this causally.

Before refinement, inject a controlled perturbation into the F1 execution-latent proposal:

```text
e_noisy = e_F1 + sigma * epsilon
```

Use isotropic Gaussian epsilon in normalized execution-latent coordinates.

Choose sigma values before evaluation from frozen training latent statistics.

Recommended preregistered scales:

```text
0.05 × train execution std
0.10 × train execution std
0.20 × train execution std
```

These are proposal perturbations, not physical-state perturbations.

This is appropriate because the current model does not observe physical state.

Evaluate:

```text
F1_noisy
F2_after_noisy_F1
random_refinement_after_noisy_F1
negative_refinement_after_noisy_F1
```

Questions:

1. Does F2 recover latent accuracy?
2. Does F2 recover decoded-action accuracy?
3. Does F2 recover closed-loop continuation success?
4. Does the recovery margin increase with perturbation scale before eventual failure?
5. Does F2 have a larger empirical basin of attraction than random/shuffled controls?

---

# 12. Closed-Loop Horizons

Use exact latent horizons:

```text
H1
H2
H4
H8
```

At CALVIN 30 Hz and 16 frames per latent step:

```text
H1 ≈ 0.533 s
H2 ≈ 1.067 s
H4 ≈ 2.133 s
H8 ≈ 4.267 s
```

Also evaluate:

```text
until task completion or source-episode end
```

as a separate continuation-success endpoint where valid.

Do not pad.
Do not cross simulator reset boundaries.

---

# 13. Primary Embodied Endpoints

The most important weakness of the current paper is the lack of behavior-level validation.

Therefore the primary endpoint is no longer latent MSE.

## P1 — Continuation task success

For each branch rollout:

```text
success = official CALVIN task predicate reached within allowed continuation horizon
```

Use the exact official evaluator.

Report:

- success rate by method;
- by task;
- by branch fraction;
- by contact phase;
- by horizon;
- by source episode.

Primary comparison:

```text
F2 vs F1
```

Use paired episode-level statistics.

## P2 — Time-to-failure / time-to-success

Where meaningful, report:

- first task success time;
- first irreversible task failure if definable;
- horizon survived before large trajectory divergence.

Do not invent a task-failure predicate without source support.

## P3 — Physical trajectory deviation

Compare executed model trajectory to source continuation using:

- joint q distance;
- TCP position distance;
- TCP orientation distance;
- gripper state disagreement;
- object pose distance;
- task-relevant object-state deviation.

Do not imply expert imitation is always the only successful path.
Treat these as secondary diagnostics.

---

# 14. Latent and Action Diagnostics During Embodied Execution

Retain the offline metrics so embodied behavior can be mechanistically connected back to previous waves.

At every latent step record:

- predicted execution latent;
- F2 intermediate refinement states;
- decoded 16-frame actions;
- execution kNN radius;
- full-latent kNN radius;
- local-PCA normal distance;
- correction-target cosine where a ground-truth source continuation exists;
- decoded-action error relative to source continuation;
- action saturation/clipping rate;
- gripper disagreement;
- actual simulator state after action execution.

This allows analysis of:

```text
latent stabilization
-> decoded command quality
-> physical trajectory
-> task success
```

---

# 15. Embodied Manifold-Success Link

The current paper claims empirical manifold stabilization.

Wave 18 must test whether the manifold metrics actually predict behavior.

For every rollout/branch compute:

```text
mean execution kNN radius
max execution kNN radius
mean local-PCA normal distance
max local-PCA normal distance
```

Relate these to:

```text
task success
decoded action error
trajectory deviation
```

Primary descriptive questions:

1. Are successful continuations associated with lower off-manifold drift?
2. Does F2 reduce off-manifold drift primarily on trajectories where it also improves behavior?
3. Are there low-kNN but failed rollouts?
4. Are there high-kNN but successful rollouts?

Do not treat kNN radius as a perfect success certificate.

---

# 16. Correction-to-Outcome Association Analysis

Do a preregistered mechanistic association analysis.

For each branch, define:

```text
Delta_error = F1 latent error - F2 latent error
Delta_decoded = F1 decoded error - F2 decoded error
Delta_knn = F1 kNN radius - F2 kNN radius
Delta_success = success_F2 - success_F1
```

Evaluate whether:

```text
larger correction-target alignment
larger kNN reduction
larger decoded-action improvement
```

are associated with larger embodied outcome improvement.

Use:

- logistic regression for success;
- rank correlation for continuous diagnostics;
- clustered bootstrap by source episode.

Do not claim causal mediation unless assumptions are justified.

Call this:

```text
mechanism-outcome association
```

---

# 17. Paired Statistical Protocol

All primary F1/F2 comparisons must be paired by the exact same source episode and branch state.

Do not treat multiple branch points from one episode as independent.

Use source episode as the highest-level cluster.

For success:

- paired bootstrap by source episode;
- report absolute success difference;
- report relative improvement;
- 95% CI;
- McNemar-type paired binary test if appropriate.

For continuous outcomes:

- paired clustered bootstrap;
- 10,000 replicates;
- 95% CI.

Predefine:

```text
seed = 180817
bootstrap_replicates = 10000
cluster_unit = source_episode
```

Do not choose seed after seeing results.

---

# 18. Primary Closed-Loop Gate

A strong embodied-validation claim is authorized only if ALL are true:

```text
G1:
F2 continuation success > F1
with paired clustered 95% CI excluding 0

G2:
F2 H4 physical/decoded rollout error < F1

G3:
F2 H8 physical/decoded rollout error < F1

G4:
F2 closed-loop execution kNN radius < F1

G5:
F2 beats RANDOM_NORM_MATCHED_REFINEMENT

G6:
F2 beats SHUFFLED_LEARNED_DIRECTION

G7:
NEGATIVE_REFINEMENT is worse than F2

G8:
frozen model hashes unchanged

G9:
zero-intervention reconstruction gate passed

G10:
no future action/state/task-label leakage
```

Do not relax this gate after seeing results.

---

# 19. Mechanism Gate for Learned Correction Direction

A stronger causal-refinement claim requires:

```text
M1:
learned F2 > norm-matched random correction

M2:
learned F2 > direction-shuffled learned correction

M3:
negative correction degrades performance relative to learned F2

M4:
correction-target cosine is positive on average

M5:
F2 proposal perturbation recovery > F1_noisy

M6:
F2 proposal perturbation recovery > random/shuffled controls

M7:
improvement in kNN/decoded error is associated with embodied success improvement
```

If the primary F2/F1 gate passes but these mechanism controls fail:

```text
closed_loop_refinement_benefit = SUPPORTED
learned_direction_mechanism = NOT_SUPPORTED
```

Do not overclaim direction-specific causality.

---

# 20. Generalization Stratification

Report the closed-loop effect separately by:

```text
task
branch fraction
contact phase
source episode duration
language-boundary crossing
initial F1 kNN radius
initial proposal error
```

The key reviewer question is:

> Is the improvement broad, or driven by one easy subset?

A positive paper claim should not depend entirely on one task.

Predefine a minimum breadth rule:

```text
F2-F1 success difference nonnegative on at least 5/6 tasks
and positive on at least 4/6 tasks
```

If a task lacks adequate embodied samples, report it separately.

---

# 21. Failure Taxonomy

For every failed closed-loop rollout, classify failure using preregistered observable categories:

```text
wrong reach direction
premature gripper open/close
insufficient contact
loss of grasp
object overshoot
task-switch mismatch
action saturation
off-manifold latent excursion
simulator/controller failure
other
```

Do not use post-hoc categories that depend on whether F1/F2 wins.

Two annotators if feasible.

Report:

- failure frequency by method;
- failure transition matrix F1 -> F2;
- which failure modes refinement repairs;
- which remain unresolved.

This is valuable for the Analysis section.

---

# 22. Optional OOD Continuation Stress Test

Only after all primary results and manifests are frozen.

If time permits, create a secondary robustness block using CALVIN-supported variations that preserve the same action interface:

- mild initial robot configuration variation;
- mild object position variation;
- language paraphrase variation;
- scene configuration variation.

Important:

Because the current model is action-history based and does not observe state directly, do not use aggressive physical perturbations and then interpret inability to recover as a refinement failure.

This block is secondary.

Primary wave-18 evidence remains matched continuation from reconstructable branch states.

---

# 23. Optional Real-Robot Sanity Check

This is bonus evidence only.
Do not block the paper on it.

If the local Franka can execute the exact same 7-D relative TCP action convention safely:

1. choose 2 simple tasks with low collision risk;
2. collect a small number of expert/reference continuations;
3. warm-start from the same causal action history;
4. compare F1 vs F2 continuation for 2–4 latent steps;
5. use conservative workspace/action clipping;
6. report qualitative/quantitative evidence separately.

Do not merge real-robot samples into the CALVIN statistical test.
Do not claim cross-embodiment transfer because the robot is still Franka.

---

# 24. Paper Claim State After Wave 18

Write:

`wave18_claim_decision.json`

Preserve:

```text
C1 = SUPPORTED
C2 = SUPPORTED
C3a_full_DEL = REJECTED
C3b_exec_DEL = REJECTED
C3c_local = STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION
C3c_long = SUPPORTED
C3d_empirical_manifold_stabilization = SUPPORTED_WITH_NUANCE
```

Add:

```text
C4_closed_loop_embodied_refinement =
SUPPORTED
or
REJECTED
or
NOT_TESTED_RECONSTRUCTION_FAILURE

C5_learned_refinement_direction_causal_value =
SUPPORTED
or
NOT_SUPPORTED
or
NOT_TESTED

C6_proposal_perturbation_recovery =
SUPPORTED
or
NOT_SUPPORTED
```

---

# 25. Paper Story Rules

## If C4 and C5 pass

Use the strongest current story:

> **Language grounds meaningful and executable action coordinates. A free predictor advances the latent trajectory, while a learned iterative correction field suppresses accumulated drift, keeps predictions near executable regions, and improves closed-loop robot continuation.**

Short version:

> **Language anchors action meaning; learned refinement stabilizes latent evolution and closed-loop execution.**

Chinese:

> **语言为动作 latent 提供语义与可执行坐标锚点；自由预测负责推进，而学习到的迭代修正场抑制累积漂移，使 latent 保持在经验可执行区域附近，并进一步提升真实闭环 continuation。**

## If C4 passes but C5 fails

Use:

> **Matched iterative refinement improves closed-loop continuation, but current evidence does not isolate a unique learned correction direction as the sole mechanism.**

Do not overclaim a causal vector field.

## If C4 fails

Do not rewrite previous offline results.

Use:

> **Refinement robustly improves offline long-horizon latent dynamics, but the improvement does not yet translate into reliable embodied continuation.**

That is still scientifically valid.

---

# 26. Required Figures

Generate publication-quality source data for these figures.
Do not spend time on final styling if the pipeline is not ready, but produce exact CSV/JSON.

## Figure A — Closed-loop success

Task-wise F1 vs F2 success with episode-clustered confidence intervals.

## Figure B — Horizon growth

H1/H2/H4/H8:

```text
latent error
decoded error
physical q/TCP deviation
kNN radius
```

for F1/F2.

## Figure C — Causal intervention

Compare:

```text
F1
F2
random norm-matched
shuffled direction
negative refinement
```

on success and decoded error.

## Figure D — Proposal perturbation recovery

Performance vs perturbation sigma.

## Figure E — Refinement trajectory

Iteration 0->4:

```text
latent error
decoded error
kNN radius
physical continuation outcome
```

## Figure F — Mechanism-outcome relation

kNN reduction / correction alignment vs embodied success improvement.

---

# 27. Required Tests

Add tests for:

- model hash freeze;
- decoder hash freeze;
- exact state reconstruction;
- deterministic twin replay;
- no branch crossing reset;
- no future action access;
- no future robot-state access;
- no future task-label access in primary protocol;
- exact H16 chunking;
- action preprocessing parity;
- F2 initialized from exact F1 proposal;
- exact four refinement iterations;
- random-control norm matching;
- shuffled-control provenance;
- negative correction sign;
- perturbation sigma computed from frozen train stats;
- source-episode clustered bootstrap;
- official success predicate use;
- all simulator outputs finite;
- no optimizer/backward calls;
- historical DEL artifacts unchanged.

Target:

```text
all tests pass
```

Do not weaken existing tests.

---

# 28. Required Deliverables

Produce:

- `eighteenth_wave_results.md`
- `eighteenth_wave_next_experiment.md`
- `wave18_frozen_model_manifest.json`
- `calvin_closed_loop_state_audit.md`
- exact simulator reconstruction code
- zero-intervention twin-replay report
- reconstruction gate JSON
- branch-source manifest
- branch-point preregistration
- causal warm-start audit
- primary closed-loop rollout logs
- F1/F2 success table
- task-wise success table
- branch-fraction success table
- contact-phase analysis
- H1/H2/H4/H8 embodied metrics
- decoded-action metrics
- q/TCP/object trajectory deviation metrics
- off-manifold metrics
- refinement intermediate-state logs
- random norm-matched control
- shuffled-direction control
- negative-refinement control
- causal-intervention result table
- proposal-perturbation preregistration
- perturbation-recovery result table
- mechanism-outcome association
- failure taxonomy report
- paired episode-clustered bootstrap
- statistical test report
- publication-figure CSV/JSON
- final claim decision JSON
- exact commands
- environment/provenance
- files-changed report
- full tests
- updated `RESEARCH_LOG.md`
- updated `NEXT_EXPERIMENT.md`

---

# 29. Final Report Must Answer These Questions

1. Can the CALVIN simulator branch state be reconstructed deterministically enough for a valid continuation experiment?
2. How many independent source episodes and branch points were used?
3. Were branch points selected before F1/F2 outputs were inspected?
4. Were representation/F1/F2/decoder completely frozen?
5. Was any future action, state, or task annotation leaked?
6. Does F2 improve official task continuation success relative to F1?
7. What is the paired episode-clustered confidence interval for the success difference?
8. Does the success advantage hold across most tasks?
9. Does F2 reduce H4 physical/decoded trajectory error?
10. Does F2 reduce H8 physical/decoded trajectory error?
11. Does F2 reduce embodied off-manifold drift?
12. Does F2 outperform norm-matched random refinement?
13. Does F2 outperform shuffled learned correction directions?
14. Does negative refinement degrade performance?
15. Does F2 recover from controlled latent proposal perturbations?
16. How large is the perturbation basin before F2 recovery fails?
17. Are correction-target alignment and kNN reduction associated with embodied success improvement?
18. Which failure modes are repaired by refinement?
19. Which failure modes remain?
20. Is C4 closed-loop embodied refinement supported?
21. Is C5 learned correction direction causal value supported?
22. Is C6 proposal perturbation recovery supported?
23. What exact paper story is now scientifically defensible?
24. Is any further CALVIN experiment still needed before writing/submission?
25. What is the single most important remaining experiment outside CALVIN?

---

# 30. Stop Conditions

Stop immediately and report rather than improvising if:

- deterministic branch replay cannot be established;
- required simulator state is unavailable;
- model hashes change;
- any future-action/state leakage is detected;
- official success predicates cannot be reproduced;
- closed-loop action preprocessing does not exactly match the frozen decoder interface;
- simulator/controller instability makes F1/F2 comparison invalid.

Do not silently replace the embodied experiment with another offline metric.

---

# 31. Strategic Interpretation

Wave 18 is designed to close the largest remaining gap in the paper:

```text
offline latent stability
        ↓
decoded command quality
        ↓
closed-loop physical behavior
        ↓
task-level continuation success
```

If this chain is supported under frozen models and matched causal interventions, the paper's central story becomes substantially stronger.

The goal is not to maximize the number of experiments.

The goal is to establish that the learned action coordinate system is:

```text
semantic
executable
predictable
long-horizon stable
behaviorally meaningful
```

and that the learned refinement direction contributes specifically to that stability.

Do not alter the method to chase a desired result.

Run the preregistered experiment exactly once after all gates are frozen.
