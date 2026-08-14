# Wave 23 Codex Prompt
# Goal-Specific Executable Alignment for Language-Conditioned Latent Dynamics
# Does Each Language Goal Have Its Own Decoder-Supported Target Subset?

## 0. Scientific Context

Wave 21 established a real causal language effect:

```text
full RedirectGain = 0.250126
95% CI = [0.136495, 0.370798]

execution RedirectGain = 0.183855
95% CI = [0.100917, 0.263777]
```

Changing only the next-language tensor from the same current latent changed the future latent trajectory in a target-dependent direction.

However:

```text
C7 = REJECTED
C8 = REJECTED
```

because the resulting trajectories did not satisfy all executable-target criteria.

Wave 22 then tested whether the missing mechanism was simple encoder-decoder consistency.

The diagnosis found:

```text
Wave21 H4 cycle residual = 2.800412
held-out GT H4 cycle residual = 0.693765

frozen cycle projection:
2.939692 -> 0.272356 residual
```

but this did NOT preserve the registered language effect:

```text
full RedirectGain after cycle4 = 0.094419
95% CI = [-0.000694, 0.171879]

endpoint accuracy:
0.516260 -> 0.401423
```

Therefore:

```text
M0_decoder_consistency_mechanism = REJECTED
C9 = NOT_TESTED
C10 = NOT_TESTED
```

The critical scientific interpretation is:

> A globally decoder-supported latent subset is not necessarily aligned with a language-selected target region.

Wave 23 must test a new, narrower mechanism:

> **Each atomic language goal may correspond to a goal-specific executable subset inside the broader decoder-supported latent space. Language redirection succeeds at selecting a direction, but fails because that direction is not aligned with the target goal's supported executable coordinates.**

The new question is:

> **Can a state-dependent language-conditioned transition be aligned to the goal-specific executable subset without collapsing to a language prototype, losing current-state dependence, or destroying transition continuity?**

This is the only primary mechanism in Wave 23.

Do NOT rescue LCT-CC.
Do NOT reopen DEL.
Do NOT add old F2 refinement.
Do NOT apply generic cycle projection.
Do NOT open a new closed-loop experiment in this wave.

---

# 1. Preserve Historical Claims

Keep unchanged:

```text
CALVIN semantic addressability = SUPPORTED
CALVIN action decodability = SUPPORTED

CALVIN local/long refinement evidence = historical supported result

DEL = REJECTED

Wave21:
language changes future latent direction = supported component
execution-space redirection = supported component
C7 = REJECTED
C8 = REJECTED

Wave22:
cycle drift = real
cycle drift associated with error = real
global cycle projection preserves executability imperfectly
M0 = REJECTED
C9/C10 = NOT_TESTED
```

Do not rewrite any historical gate.

---

# 2. Frozen Assets

Freeze and hash:

```text
Wave21 CALVIN encoder
Wave21 decoder
Wave21 semantic projection
Wave21 text encoder
Wave21 B1 LCT
normalization statistics
Wave21 train/dev/test session split
Wave21 transition inventory
```

Write:

`wave23_frozen_manifest.json`

Required:

```text
representation optimizer steps = 0
encoder optimizer steps = 0
decoder optimizer steps = 0
text encoder optimizer steps = 0
```

Wave23 may train only the new transition/alignment model defined below.

---

# 3. Data

Reuse the exact Wave21 physically continuous transition dataset:

```text
560 annotation-onset transitions
31 source sessions
6 atomic next-goal classes
```

Keep the exact:

```text
train/dev/test source-session split
H=16 chunking
sparse-annotation disclosure
next-annotation-start boundary definition
```

Do not fill annotation gaps with guessed labels.

Do not concatenate sessions.

Do not cross resets.

---

# 4. Goal-Specific Executable Support Sets

This is the key new object.

For each atomic goal `g`, define a train-only support set:

```text
S_g
```

using frozen training latents whose action annotation is `g`.

Do NOT use Wave23 predictions to define S_g.

For every train latent z in goal g, compute:

```text
cycle residual r_cycle(z) = ||E(D(z)) - z||

execution kNN density

decoder reconstruction error

semantic similarity to goal language
```

Use these only to characterize the supported set.

---

# 5. Supported Goal Core

Define a conservative train-only goal core:

```text
C_g ⊂ S_g
```

using a frozen development-independent rule:

Primary rule:

```text
C_g =
goal-g train latents
whose cycle residual <=
the 75th percentile of cycle residual
among goal-g train latents
```

Do not tune this percentile after seeing development outcomes.

Also record:

```text
50th percentile core
90th percentile core
```

for descriptive sensitivity only.

Primary inference must use 75%.

Write:

`wave23_goal_core_manifest.json`

---

# 6. Why Goal-Specific Instead of Global Projection?

Wave22 showed that generic cycle projection reduced residual but also reduced target identity.

Wave23 must explicitly quantify:

```text
distance from Wave21 endpoint to global supported set
distance to requested goal core C_g
distance to competing goal cores C_h
```

For each prediction z_hat(g), compute:

```text
d_global
d_goal
min_competing_goal_distance
goal_margin =
min_competing_goal_distance - d_goal
```

Primary diagnostic hypothesis:

```text
Wave21 predictions may be globally supportable
yet still poorly aligned to C_g
```

This would explain Wave22.

---

# 7. Phase A — Frozen Goal/Support Geometry Diagnosis

Before any optimizer step, use only train + development.

For every development transition:

1. obtain frozen Wave21 LCT prediction;
2. compute goal-core distances;
3. compute global cycle residual;
4. compute target-region identity;
5. compute decoder/re-encoder identity;
6. compute continuity.

Test:

```text
D1:
requested goal-core distance predicts endpoint target failure

D2:
requested goal-core margin predicts target identity

D3:
global cycle residual alone explains less target identity
than goal-core margin

D4:
Wave22 cycle projection often moves toward global support
while moving away from the requested goal core

D5:
execution-space goal-core geometry shows the same pattern
```

Use source-session clustered bootstrap.

---

# 8. Mechanism Authorization Gate M1

Define:

```text
M1_goal_specific_executable_alignment
```

SUPPORTED_FOR_INTERVENTION only if all are true:

```text
A1:
goal-core margin is positively associated with endpoint correctness

A2:
goal-core distance is positively associated with decoded-action error

A3:
goal-core margin explains endpoint identity beyond global cycle residual
in a preregistered regression / partial association

A4:
Wave22 global cycle projection reduces cycle residual
but decreases requested goal-core margin on average
or on a substantial preregistered fraction

A5:
the same effect exists in execution dimensions
```

If M1 fails:

STOP.

Do not train a goal-alignment model.

Write:

`wave23_goal_alignment_mechanism_rejected.md`

---

# 9. New Model: Goal-Aligned Language-Conditioned Transition

Only if M1 passes.

Call:

```text
LCT-GA
```

Start from the exact Wave21 B1 architecture and training setup.

Train a new model on the same Wave21 train split.

The model still predicts the observed future latent trajectory from:

```text
z_previous
z_current
next_language_embedding
```

No observation/state input is added.

---

# 10. Single New Alignment Term

Primary objective:

```text
L_total =
L_latent_prediction
+
lambda_decode * L_decoded_action
+
lambda_align * L_goal_executable_alignment
```

The new term is the ONLY new scientific factor.

Define:

```text
L_goal_executable_alignment =
soft distance from predicted latent
to the requested goal core C_g
```

Preferred differentiable implementation:

```text
softmin over K nearest goal-core train latents
```

or a frozen differentiable kernel density surrogate.

Do NOT use a simple goal prototype as the target.

Do NOT minimize distance to the mean of C_g.

The alignment term must preserve a local set, not a centroid.

---

# 11. State-Conditioned Local Alignment

To prevent prototype collapse, do NOT align to all of C_g equally.

For current state z_t:

1. identify K candidate goal-core latents in C_g;
2. weight candidates using similarity between their source-preceding latent and z_t where source transition metadata exists;
3. otherwise use nearest goal-core execution latents.

Primary frozen K:

```text
K = 20
```

Do not sweep K.

This defines a state-conditioned target neighborhood:

```text
N_g(z_t)
```

The alignment loss is toward the local supported neighborhood, not the global class mean.

---

# 12. No Explicit Classification Loss

Do NOT add:

```text
cross-entropy goal classification
endpoint label loss
prototype classification loss
semantic retrieval loss
```

to LCT-GA.

The experiment must test geometry alignment, not train an endpoint classifier and then evaluate endpoint classification.

---

# 13. No Cycle Loss

Do NOT include:

```text
||E(D(z))-z||
```

as a training loss in the primary Wave23 model.

Wave22 rejected pure cycle consistency as the authorized mechanism.

Cycle residual remains an evaluation diagnostic only.

---

# 14. lambda_align Selection

Development-only candidate set:

```text
lambda_align ∈ {0.03, 0.1, 0.3}
```

Freeze this set before training.

Selection rule:

choose the smallest lambda satisfying all development conditions:

```text
RedirectGain >= 90% of Wave21 B1 development RedirectGain

Execution RedirectGain >= 90% of Wave21 B1 development execution RedirectGain

endpoint macro accuracy >= Wave21 B1 + 0.05

decoded/reencoded target accuracy >= Wave21 B1 + 0.05

H4 decoded action MSE no worse than Wave21 B1 by >5%

continuity no worse than Wave21 B1
```

If none pass:

STOP.

Do not add another lambda.

---

# 15. Model Seeds

Use exactly six preregistered seeds.

Write:

`wave23_seed_preregistration.json`

No seed replacement.

No extra seeds after evaluation.

---

# 16. Baselines

Required:

```text
B0 = Wave21 unconditional transition
B1 = frozen Wave21 correct-language LCT
P  = language prototype
C4 = Wave22 frozen cycle4 diagnostic
GA = new LCT-GA
```

Optional descriptive:

```text
shuffled-language Wave21 B2
null-language Wave21 B3
```

Do NOT train new versions of old baselines unless required for exact pairing.

---

# 17. Same-State Six-Way Counterfactual Test

Repeat the exact Wave21 intervention:

for the identical current state z_t, run:

```text
GA(z_t, A0)
GA(z_t, A1)
GA(z_t, A2)
GA(z_t, A3)
GA(z_t, A4)
GA(z_t, A5)
```

Only language changes.

Store all trajectories.

Primary metrics:

```text
RedirectGain
Execution RedirectGain
endpoint macro accuracy
decoded/reencoded endpoint accuracy
goal-core margin
execution goal-core margin
```

---

# 18. Main Question

Wave23 does NOT ask:

```text
Can language change the vector field?
```

That component already passed in Wave21.

Wave23 asks:

> **Can the language-changed vector field terminate in the correct goal-specific executable subset while preserving state-dependent transition structure?**

---

# 19. Current-State Dependence Gate

LCT-GA must not collapse to goal-conditioned retrieval.

Compare against:

```text
language prototype
goal-core nearest-neighbor retrieval
goal-core mean
```

on held-out transitions.

Require GA to beat language-only controls on both:

```text
H2 full-latent MSE
H4 decoded action MSE
```

Also report:

```text
within-goal endpoint variance
current-state / endpoint residual correlation
decoded-action diversity
```

---

# 20. Transition Continuity

Measure exactly as Wave21:

```text
latent boundary jump
execution boundary jump
decoded action discontinuity
velocity direction change
```

Compare:

```text
Wave21 B1
Wave22 cycle4
prototype
LCT-GA
ground truth
```

The new model must not obtain target identity by teleporting to a supported target point.

---

# 21. Decoder Consistency as Diagnostic

Even though no cycle loss is used, evaluate:

```text
cycle residual
semantic cycle residual
execution cycle residual
```

Question:

> Does aligning to goal-specific supported coordinates naturally reduce cycle drift without explicitly optimizing cycle consistency?

This would be strong evidence for the geometry interpretation.

Do not use cycle residual as the primary optimization target.

---

# 22. Goal-Core Identity

For every endpoint classify by nearest train-only goal core.

Report:

```text
full latent goal-core Top-1
execution-only goal-core Top-1
decoded/reencoded goal-core Top-1
mean target rank
goal-core margin
```

Primary accuracy threshold remains:

```text
>= 0.60
```

Do not lower it.

---

# 23. Primary Claim C11

Define:

```text
C11_goal_specific_executable_alignment
```

SUPPORTED only if ALL are true on held-out source sessions:

## G1 — Full language redirection preserved

```text
RedirectGain > 0
clustered lower95 > 0

and
GA RedirectGain >= 0.90 * Wave21 B1 RedirectGain
```

## G2 — Execution redirection preserved

```text
Execution RedirectGain > 0
clustered lower95 > 0

and
GA execution RedirectGain >= 0.90 * Wave21 B1 execution RedirectGain
```

## G3 — Endpoint identity repaired

```text
full endpoint macro accuracy >= 0.60
```

## G4 — Decode/reencode identity repaired

```text
decoded/reencoded macro accuracy >= 0.60
```

## G5 — Current state matters

GA beats both:

```text
language prototype
goal-core retrieval
```

on:

```text
H2 full MSE
H4 decoded action MSE
```

## G6 — Continuity

GA continuity must be:

```text
better than language prototype
and
no worse than Wave21 B1
```

## G7 — No catastrophic decoder inconsistency

Held-out cycle error must be:

```text
lower than Wave21 B1
```

with favorable clustered CI.

It is not required to reach Wave22's rejected hard cycle tolerance.

## G8 — Breadth

Positive goal-core target margin for at least:

```text
5/6 atomic goal classes
```

---

# 24. Stronger Claim C12

Define:

```text
C12_language_as_goal_specific_executable_coordinate
```

SUPPORTED only if C11 passes and:

1. same-state six-way endpoint identity >=0.65;
2. execution-only endpoint identity >=0.65;
3. decoded/reencoded identity >=0.60;
4. current-state dependence passes;
5. continuity passes;
6. `lift_blue_block_slider -> place_in_slider` is positive on target margin and decoded action error;
7. no single goal contributes >40% of aggregate RedirectGain.

Safe wording:

> **Language specifies a goal-specific executable subset of the action-coordinate space. From the same current latent state, changing only the next atomic language goal redirects the transition toward a different supported executable region while preserving state-dependent continuity.**

---

# 25. Canonical Lift-to-Place Case

Primary case study:

```text
lift_blue_block_slider -> place_in_slider
```

Wave22 diagnostic cycle projection worsened target distance for this pair.

Wave23 must compare:

```text
Wave21 B1
Wave22 cycle4
prototype
LCT-GA
ground truth
```

Plot:

```text
goal-core distance
goal-core margin
execution goal-core distance
decoded action MSE
cycle residual
continuity
```

Use all eligible held-out cases.

Do not cherry-pick.

---

# 26. Goal-Specific Geometry Visualization

For each goal:

fit visualization only on train latents.

Show:

```text
all goal latents
goal executable core C_g
Wave21 endpoint
Wave22 cycle4 endpoint
Wave23 GA endpoint
ground truth
```

Use:

```text
PCA or UMAP
```

for visualization only.

No 2-D metric enters any claim gate.

---

# 27. Pairwise Goal Geometry

For all goal pairs g,h report:

```text
distance(C_g, C_h)
overlap rate
nearest-neighbor confusion
semantic cosine
execution-space separation
```

Purpose:

Some goals may share executable regions.

This may explain why 0.60 six-way identity is difficult.

Do not change the threshold after seeing overlap.

Report honestly.

---

# 28. Contact / Physical-Phase Secondary Analysis

If source metadata permits, stratify transitions by:

```text
free motion
pre-contact
contact
transport
release/post-contact
```

This is descriptive only.

Do not redefine primary samples.

---

# 29. Statistical Protocol

Independent unit:

```text
continuous source session
```

Use:

```text
10,000 bootstrap replicates
cluster = source session
seed = 230823
```

Same-state language swaps paired within boundary.

Use next-goal stratification when appropriate.

---

# 30. Held-Out Discipline

Before held-out inference freeze:

```text
M1 decision
selected lambda_align
six seed list
checkpoint selection
goal-core manifests
K=20
all thresholds
all metrics
bootstrap seed
claim gates
```

Write:

`wave23_final_test_preregistration.json`

Evaluate held-out once.

No post-test retraining.

---

# 31. Required Mechanistic Regressions

On development and then frozen held-out analysis:

Model endpoint correctness / target margin using:

```text
global cycle residual
goal-core distance
goal-core margin
execution goal-core margin
current-state distance
```

Primary comparison:

Does goal-specific executable geometry explain target identity beyond global cycle consistency?

Report:

```text
coefficient signs
standardized effects
clustered CI
incremental R^2 / likelihood improvement
```

Do not overclaim causal mediation.

---

# 32. Required Figures

## Figure 1
Global support vs goal-specific support schematic.

## Figure 2
Wave22 cycle projection:

```text
cycle residual improves
goal-core margin worsens
```

## Figure 3
Goal-core margin vs endpoint correctness.

## Figure 4
Wave21 B1 vs Wave22 cycle4 vs Wave23 GA.

## Figure 5
Same-state six-way language redirection into goal-specific cores.

## Figure 6
Decode/reencode identity.

## Figure 7
Lift -> place case.

---

# 33. Required Tables

### Table A
Goal-core statistics per atomic action.

### Table B
Phase-A geometry diagnosis.

### Table C
Main held-out comparison:

```text
B1
cycle4
prototype
GA
```

Rows:

```text
RedirectGain
Execution RedirectGain
endpoint accuracy
decoded/reencoded accuracy
H2 MSE
H4 decoded MSE
goal-core margin
cycle residual
continuity
```

### Table D
Per-goal results.

### Table E
C11/C12 gate table.

---

# 34. Failure Taxonomy

Use fixed categories:

```text
goal-core misalignment
global-support / target-support conflict
semantic-execution mismatch
prototype collapse
current-state ignored
decoder inconsistency
goal overlap ambiguity
continuity failure
goal-specific failure
long-horizon accumulation
other
```

---

# 35. Required Tests

At minimum:

```text
Wave21 representation hashes unchanged
Wave21 decoder hashes unchanged
Wave21 split unchanged
Wave21 transition inventory unchanged

goal cores use TRAIN only
75th percentile rule exact
K = 20 exact

no held-out data in goal-core construction
no held-out data in lambda selection

no target classification loss
no prototype loss
no cycle loss
no F2 refinement
no DEL

same-state six-way intervention changes only language

LCT-GA receives current state and language
prototype controls do not receive current state

source-session bootstrap
10000 replicates
seed 230823

all outputs finite
all JSON valid
```

Target:

```text
all tests pass
```

---

# 36. Stop Conditions

STOP if:

```text
M1 goal-specific alignment mechanism fails

goal cores cannot be defined for >=4 actions

goal-core construction accidentally uses test data

no lambda_align passes development preservation rules

full RedirectGain collapses

execution RedirectGain collapses

current-state dependence fails on development

held-out data is opened before preregistration freeze
```

If C11 fails on held-out:

do not add another rescue mechanism in Wave23.

---

# 37. Required Deliverables

Produce:

```text
twenty_third_wave_results.md
twenty_third_wave_next_experiment.md

wave23_frozen_manifest.json
wave23_goal_core_manifest.json

wave23_phaseA_goal_geometry.md
wave23_goal_core_association_report.md
wave23_mechanism_gate.json

wave23_model_preregistration.json
wave23_seed_preregistration.json
wave23_alignment_weight_selection.json
wave23_final_test_preregistration.json

wave23_training_report.md
wave23_statistical_report.md

wave23_main_comparison.md
wave23_same_state_language_swap.md
wave23_decode_reencode_results.md
wave23_continuity_results.md
wave23_goal_geometry_analysis.md
wave23_lift_to_place_case.md

wave23_failure_taxonomy.md
wave23_claim_decision.json

publication_tables/
publication_figures_data/

exact_commands.sh
environment_freeze.txt
files_changed.txt
tests_report.txt

updated_RESEARCH_LOG.md
updated_NEXT_EXPERIMENT.md
```

---

# 38. Claim Decision JSON

Write:

`wave23_claim_decision.json`

with:

```text
M1_goal_specific_executable_alignment:
SUPPORTED_FOR_INTERVENTION / REJECTED

C11_goal_specific_executable_alignment:
SUPPORTED / REJECTED / NOT_TESTED

C12_language_as_goal_specific_executable_coordinate:
SUPPORTED / REJECTED / NOT_TESTED

full_redirect_preserved:
true / false / inconclusive

execution_redirect_preserved:
true / false / inconclusive

endpoint_identity_repaired:
true / false / inconclusive

decode_reencode_identity_repaired:
true / false / inconclusive

current_state_matters:
true / false / inconclusive

continuity_preserved:
true / false / inconclusive

cycle_error_reduced_without_cycle_loss:
true / false / inconclusive
```

---

# 39. Final Report Questions

The final report must answer:

1. Are the six action goals associated with distinct train-only executable cores?
2. How much do those cores overlap?
3. Does requested goal-core distance predict endpoint failure?
4. Does goal-core margin predict endpoint identity?
5. Does it explain target identity beyond global cycle residual?
6. Did Wave22 cycle projection move toward global support but away from the requested goal core?
7. Does this happen in execution dimensions?
8. Was M1 authorized?
9. Which lambda_align was selected?
10. Did LCT-GA preserve full RedirectGain?
11. Did it preserve execution RedirectGain?
12. Did endpoint macro accuracy reach >=0.60?
13. Did decoded/reencoded accuracy reach >=0.60?
14. Did GA beat language prototype on H2 full MSE?
15. Did GA beat prototype on H4 decoded MSE?
16. Did current state still matter?
17. Did continuity improve?
18. Did cycle residual improve without a cycle loss?
19. Did at least 5/6 target actions show positive target margins?
20. Did lift_blue_block_slider -> place_in_slider improve?
21. Is C11 supported?
22. Is C12 supported?
23. What exact mechanism now best explains Wave21 and Wave22?
24. What exact paper claim is defensible?
25. If C11 passes, what closed-loop experiment should follow?
26. If C11 fails, what part of the language-target-coordinate hypothesis remains supported?

---

# 40. Interpretation Rules

If M1 and C11/C12 pass:

> **Wave21 showed that language changes the latent vector field, while Wave22 showed that global decoder support is not sufficient to preserve the selected target. Wave23 demonstrates that each language goal corresponds to a goal-specific executable subset: aligning the transition to this local supported set preserves causal language redirection while restoring target identity and decodability.**

Chinese:

> **Wave21 已经证明语言能够改变 latent dynamics 的方向；Wave22 进一步说明，全局地把 latent 拉回 decoder-supported set 会损失目标语义。Wave23 若通过，则说明真正需要的是 goal-specific executable alignment：每个语言目标对应可执行空间中的一个局部支持区域，语言负责选择目标方向，当前 latent 决定从哪里出发，而 transition model 负责在保持连续性的同时进入对应的可执行目标区域。**

If goal-core alignment helps identity but current-state dependence fails:

> Language selects executable action regions, but state-dependent transition dynamics are not yet established.

If current-state dependence holds but endpoint identity remains <0.60:

> Language causally redirects state-dependent dynamics, but the six-way action regions are not sufficiently separable for a target-coordinate claim.

If M1 fails:

> The Wave21/Wave22 failure cannot be explained by global-vs-goal-specific executable geometry, so do not pursue alignment losses further.

---

# 41. Strategic Meaning

The project now has a progressively sharper sequence of findings:

```text
1. language anchors action meaning

2. action latents remain decodable

3. language changes future latent direction

4. execution dimensions also change

5. unrestricted language-conditioned trajectories drift

6. global decoder-consistency projection repairs support
   but harms target identity

7. Wave23 asks whether the missing structure is
   goal-specific executable geometry
```

The conceptual model is:

```text
current latent z_t
        +
next language goal g
        ↓
language selects a target executable subset C_g
        ↓
state-dependent transition chooses a path
from z_t toward C_g
        ↓
decoder produces continuous robot actions
```

This is a cleaner hypothesis than treating the entire decoder-supported space as one global manifold.

That is the core Wave23 experiment.
