# Wave 24 Codex Prompt
# State- and Horizon-Conditioned Transition Displacement Families
# Does Language Select a Family of Executable Displacements Rather Than a Static Endpoint Region?

## 0. Scientific Context

The project has converged to a narrower and stronger hypothesis.

The most robust positive finding remains Wave 21:

> **Changing only the next-goal language causally redirects the predicted latent trajectory, including in execution-space coordinates.**

Frozen Wave-21 evidence:

```text
full RedirectGain = 0.250126
95% CI = [0.136495, 0.370798]

execution RedirectGain = 0.183855
95% CI = [0.100917, 0.263777]
```

This finding remains valid.

Wave 22 showed that a global decoder-consistent set is not sufficient:

```text
cycle residual:
2.939692 -> 0.272356 after frozen cycle projection

but:

full RedirectGain:
0.250126 -> 0.094419
95% CI after projection = [-0.000694, 0.171879]

endpoint accuracy:
0.516260 -> 0.401423
```

Thus:

```text
global decoder support != language-selected target geometry
```

Wave 23 then showed that goal-specific static geometry is explanatory but not corrective:

```text
goal-core margin strongly predicts endpoint identity

margin/correctness Pearson r = 0.7148
Spearman rho = 0.8442

incremental R^2 beyond global cycle residual = 0.4250
```

However, static goal-core alignment failed as an intervention:

```text
M1 = SUPPORTED_FOR_INTERVENTION

but no lambda_align candidate passed development selection

small lambda preserved:
- RedirectGain
- decoded MSE
- continuity

yet worsened:
- endpoint identity
- decode/reencode identity
```

Wave 23 therefore stopped before held-out evaluation.

This suggests the current model class is missing **transition-conditioned correspondence**.

The new hypothesis is:

> **Language does not specify a fixed executable endpoint region. Instead, language selects a state- and horizon-conditioned family of executable latent displacements. The current latent determines which displacement within that family is appropriate.**

Chinese:

> **语言不是直接指定一个固定的可执行终点区域，而是根据当前状态和时间尺度，选择一组可执行的 latent transition directions；当前 latent 决定具体应该沿其中哪一个方向演化。**

Wave 24 must test this hypothesis directly.

---

# 1. Primary Scientific Question

For a transition sample:

```text
(z_previous, z_current, goal, z_future_h)
```

with:

```text
h in {1, 2, 4}
```

define the true transition displacement:

```text
delta_true(h) = z_future_h - z_current
```

The key question is:

> **Among training transitions with the same language goal, do transitions originating from states near the current latent provide a better model of the correct future displacement than static endpoint cores or goal-only prototypes?**

More formally, for current latent `z_current`, goal `g`, and horizon `h`, does there exist a local transition family:

```text
D_{g,h}(z_current)
```

such that nearby source states with the same `(goal, h)` have displacements that predict:

```text
direction
magnitude
future latent
decoded action
target identity
```

better than static endpoint geometry?

Wave 24 begins as a pure diagnostic.

Do NOT train a new transition loss until the displacement-family mechanism is prospectively authorized.

---

# 2. Preserve Historical Results

Do not reinterpret earlier waves.

Keep:

```text
Wave21:
language causal redirection component = SUPPORTED
execution-space redirection component = SUPPORTED
C7 = REJECTED
C8 = REJECTED

Wave22:
global cycle drift = REAL
global cycle projection = NOT SUFFICIENT
M0 = REJECTED

Wave23:
goal-specific geometry = EXPLANATORY
static goal-core alignment = NOT CORRECTIVE
M1 = SUPPORTED_FOR_INTERVENTION
C11/C12 = NOT_TESTED
held-out Wave23 test = UNOPENED
```

Do not reopen:

```text
DEL
F2 rescue
cycle-consistency rescue
static goal-core attraction
closed-loop execution
```

in Wave 24.

---

# 3. Frozen Assets

Hash and freeze:

```text
Wave21 CALVIN action encoder
Wave21 decoder
Wave21 semantic projection
Wave21 text encoder
Wave21 B1 LCT
Wave21 B0 unconditional
Wave21 train/dev/test session split
Wave21 transition inventory
Wave23 train-only goal-core definitions
normalization statistics
```

Write:

`wave24_frozen_manifest.json`

Required:

```text
representation optimizer steps = 0
encoder optimizer steps = 0
decoder optimizer steps = 0
text encoder optimizer steps = 0
Wave21 LCT optimizer steps = 0 during Phase A
```

---

# 4. Dataset Reconstruction

Reconstruct paired transition records from the exact Wave21 source sessions.

Each record must contain:

```text
source_session
boundary_id
previous_goal
next_goal
z_previous
z_current

z_future_H1
z_future_H2
z_future_H4

delta_H1 = z_future_H1 - z_current
delta_H2 = z_future_H2 - z_current
delta_H4 = z_future_H4 - z_current

future_actions_H1
future_actions_H2
future_actions_H4
```

The source frames must be physically contiguous.

Do not fill annotation gaps with synthetic labels.

Do not cross reset/discontinuity boundaries.

Write:

`wave24_paired_transition_inventory.parquet`

and:

`wave24_paired_transition_inventory_report.md`

---

# 5. Horizon Definition

Use exact latent horizons:

```text
H1 = 1 latent step = 16 frames
H2 = 2 latent steps = 32 frames
H4 = 4 latent steps = 64 frames
```

Primary Wave24 analysis uses all three separately.

Do NOT collapse H1/H2/H4 into one endpoint family.

This is essential.

The hypothesis explicitly predicts:

```text
D_{g,1} != D_{g,2} != D_{g,4}
```

in general.

---

# 6. Split Discipline

Use the exact Wave21 source-session split.

No transition from the same session may cross train/dev/test.

Wave23 held-out test must remain unopened during Phase A.

Use:

```text
train = transition support construction
development = mechanism authorization
held-out = unopened until Phase B preregistration passes
```

Write:

`wave24_split_freeze.json`

---

# 7. Three Competing Support Structures

Wave 24 must compare exactly three frozen support structures.

## S1 — Static Goal Core

Wave23 structure:

```text
C_g
```

A goal-specific endpoint subset independent of horizon and source current state.

This is the known static baseline.

## S2 — Horizon-Specific Goal Core

Define:

```text
C_{g,h}
```

using train-only future endpoints for:

```text
goal = g
horizon = h
```

Thus:

```text
C_{g,1}
C_{g,2}
C_{g,4}
```

are different endpoint support sets.

## S3 — Source-Conditioned Transition Family

Define:

```text
D_{g,h}(z_current)
```

from train-only paired transitions.

Each training element contains:

```text
source current latent z_i
paired displacement delta_i(h)
```

For a development query `(z_q, g, h)`:

1. filter train transitions to same goal `g`;
2. filter to same horizon `h`;
3. retrieve nearest source current latents `z_i` to `z_q`;
4. use their paired displacement vectors `delta_i(h)`.

This is the primary new structure.

Importantly:

> Neighbor selection is based on source/current latent similarity, NOT endpoint similarity.

This corrects the Wave23 fallback error.

---

# 8. Distance Metric for Source Conditioning

Use frozen normalized latent coordinates.

Primary neighbor distance:

```text
d_source(z_q, z_i)
=
||z_q_exec - z_i_exec||_2
```

using execution dimensions only.

Also compute full-latent distance descriptively.

Primary K:

```text
K = 20
```

Do not sweep K.

If fewer than 20 same-goal/horizon train transitions exist, use all available and report count.

Minimum admissible support:

```text
>= 8 train transitions
```

for a `(goal,horizon)` cell.

Cells below this are excluded prospectively.

---

# 9. Displacement Family Statistics

For each `(goal, horizon)`:

report train-only:

```text
number of transitions
mean displacement norm
displacement covariance
effective rank
pairwise cosine distribution
pairwise magnitude distribution
within-goal variation
between-goal variation
```

The purpose is to test whether:

```text
one language goal corresponds to one fixed direction
```

or:

```text
one language goal corresponds to a state-conditioned family of directions
```

Do not use these statistics to tune the model.

---

# 10. Phase A1 — Static Endpoint vs Horizon-Specific Endpoint

On development only, compare S1 and S2.

For each query:

```text
ground-truth future endpoint z_future_h
```

measure:

```text
distance to C_g
distance to C_{g,h}

goal margin
endpoint identity
decoded action consistency
```

Question:

> Does conditioning target support on horizon explain future endpoints better than a static goal core?

Define:

```text
HorizonCoreGain =
d(z_future_h, C_g)
-
d(z_future_h, C_{g,h})
```

Positive means horizon-specific support is better.

Use source-session clustered bootstrap.

---

# 11. Phase A2 — Source-Conditioned Displacement Prediction

For each development query `(z_q, g, h)` retrieve K train source neighbors.

Each neighbor provides paired displacement:

```text
delta_i(h)
```

Construct three non-trained predictors.

### D1 — Mean paired displacement

```text
delta_hat_mean =
mean_i delta_i(h)
```

### D2 — Distance-weighted displacement

Use frozen weights:

```text
w_i = exp(-d_source(z_q,z_i)^2 / tau^2)
```

Set:

```text
tau = median K-neighbor source distance
computed from TRAIN only
```

Then:

```text
delta_hat_weighted =
sum_i w_i delta_i / sum_i w_i
```

### D3 — Nearest paired displacement

```text
delta_hat_1NN =
delta_(nearest source state)
```

Primary diagnostic uses D2.

D1/D3 are controls.

---

# 12. Predicted Endpoint from Displacement Family

For each query:

```text
z_hat_h =
z_current + delta_hat(h)
```

Evaluate against:

```text
z_future_h
```

Metrics:

```text
full latent MSE
semantic latent MSE
execution latent MSE
decoded action MSE
target-region identity
goal-core margin
horizon-core margin
cycle residual
continuity error
```

This is still diagnostic.

No learned model.

---

# 13. Displacement Direction Metrics

For development query:

```text
delta_true
delta_hat
```

compute:

```text
cosine(delta_hat, delta_true)
norm ratio = ||delta_hat|| / ||delta_true||
angular error
execution-only cosine
semantic-only cosine
```

Primary new metric:

```text
PairedDisplacementCosine
```

Question:

> Do source-conditioned same-goal transition displacements predict the actual direction better than goal-only mean displacement?

---

# 14. Compare Against Goal-Only Displacement Baseline

Define train-only:

```text
delta_goal_mean(g,h)
=
mean displacement over all train transitions
with goal g and horizon h
```

This baseline knows:

```text
goal
horizon
```

but NOT current state.

Compare against source-conditioned D2.

This is essential.

If D2 does not beat goal-only displacement, current state is not needed.

---

# 15. Compare Against Static Endpoint Baselines

Also compare:

```text
Wave23 static goal core
horizon-specific endpoint core
goal prototype
```

on reconstructed future endpoint and decoded action.

The displacement model should not be judged only by target classification.

It must predict the actual future path from the current state.

---

# 16. Mechanism Claim M2

Define:

```text
M2_state_horizon_conditioned_displacement_family
```

SUPPORTED_FOR_INTERVENTION only if ALL development gates pass.

## A1 — Horizon matters

Require:

```text
HorizonCoreGain > 0
clustered lower95 > 0
```

aggregated across eligible goals/horizons.

## A2 — Source-conditioned displacement predicts true direction

Require D2:

```text
mean full cosine > 0
lower95 > 0
```

and:

```text
mean execution cosine > 0
lower95 > 0
```

## A3 — Current state matters beyond goal+horizon

Require D2 to beat `delta_goal_mean(g,h)` on BOTH:

```text
full latent MSE
execution latent MSE
```

with favorable clustered 95% CI.

## A4 — Source-conditioned displacement beats static endpoint geometry

Require D2-derived endpoint to beat Wave23 static goal-core/prototype baseline on:

```text
H2 full latent MSE
H4 decoded action MSE
```

## A5 — Target identity is not degraded

Require D2 endpoint identity:

```text
>= Wave21 B1 endpoint identity on development
```

and decoded/reencoded identity:

```text
>= Wave21 B1 development identity
```

No +0.05 requirement yet.

Phase A is mechanism diagnosis.

## A6 — Continuity

Require D2 transition continuity:

```text
no worse than Wave21 B1
```

on development.

If any required gate fails:

```text
M2 = REJECTED
```

STOP.

Do not train a displacement-matching model.

---

# 17. Interpretation if M2 Passes

If M2 passes, the supported mechanism becomes:

> **For a fixed language goal and horizon, future latent displacement depends systematically on the current state. Nearby source states provide paired transition directions that predict future dynamics better than static goal endpoints or goal-only displacement averages.**

This directly supports:

```text
language selects a transition family
current state selects a member of that family
horizon selects the temporal scale
```

Only then proceed to Phase B.

---

# 18. Phase B — New Model: LCT-TD

Only if M2 passes.

Call:

```text
LCT-TD
```

for:

```text
Language-Conditioned Transition with Transition-Displacement Matching
```

Start from the exact Wave21 B1 architecture.

Representation remains frozen.

Decoder remains frozen.

Text encoder remains frozen.

---

# 19. Primary New Loss

The only new scientific factor is:

```text
L_transition_match
```

For a predicted future displacement:

```text
delta_pred(h)
=
z_pred_h - z_current
```

retrieve the train-only source-conditioned neighborhood:

```text
N_{g,h}(z_current)
```

with K=20 based on source current latent.

Each neighbor has paired:

```text
delta_train_j(h)
```

Define:

```text
L_transition_match
=
softmin_j ||delta_pred(h) - delta_train_j(h)||^2
```

The loss is over displacement, NOT endpoint.

This is the core Wave24 intervention.

---

# 20. Multi-Horizon Training

Use:

```text
h in {1,2,4}
```

Train the model to predict all three horizons or recursively predict and evaluate all three.

The transition-match loss must be horizon-specific:

```text
L_TM =
sum_h alpha_h * softmin_j ||delta_pred(h)-delta_train_j(h)||^2
```

Freeze:

```text
alpha_1 = alpha_2 = alpha_4 = 1
```

unless the existing architecture mathematically requires normalization.

Do not tune horizon weights.

---

# 21. Full Training Objective

Primary:

```text
L_total =
L_latent_prediction
+
lambda_decode * L_decoded_action
+
lambda_TM * L_transition_match
```

Do NOT include:

```text
endpoint attraction loss
goal-core softmin loss
cycle loss
classification loss
prototype loss
PCA normal-distance loss
F2 refinement
DEL
```

Only one new factor is allowed.

---

# 22. lambda_TM Selection

Development-only candidate set:

```text
lambda_TM in {0.03, 0.1, 0.3}
```

Freeze before training.

Selection rule:

choose the smallest lambda satisfying all development conditions:

```text
RedirectGain >= 90% of Wave21 B1
Execution RedirectGain >= 90% of Wave21 B1

H2 execution MSE < Wave21 B1
H4 decoded MSE < Wave21 B1

endpoint macro accuracy >= Wave21 B1
decode/reencode accuracy >= Wave21 B1

continuity no worse than Wave21 B1

current-state dependence gate passes
```

If none pass:

STOP.

No extra lambda.

---

# 23. Six Seeds

Use exactly six preregistered seeds.

Pair with Wave21 initialization where possible.

Write:

`wave24_seed_preregistration.json`

No replacement seeds.

No extra seeds after development evaluation.

---

# 24. Current-State Dependence Test

Wave24's central claim requires that current state matters.

For held-out query `(z_current, g, h)`, compare:

```text
LCT-TD
vs
goal+horizon mean displacement
vs
language prototype
vs
static goal core
```

Require LCT-TD to beat all goal-only baselines on:

```text
H2 full MSE
H4 decoded MSE
```

This is mandatory.

---

# 25. Same-State Language Swap

Repeat Wave21 exactly.

For same:

```text
z_previous
z_current
history
weights
seed
```

change only:

```text
next language goal
```

Run six goals.

Compute:

```text
RedirectGain
Execution RedirectGain
endpoint identity
decode/reencode identity
transition-family margin
```

This ensures transition matching does not erase causal language controllability.

---

# 26. Transition-Family Membership Metric

Define train-only family support for query `(z_current,g,h)`.

For predicted displacement:

```text
delta_pred
```

compute distance to source-conditioned paired displacement neighborhood:

```text
d_family(delta_pred)
=
mean distance to K paired delta_train
```

Also compute competing goal family distances:

```text
d_family_h
```

Define:

```text
family_margin =
min_{wrong goal} d_wrong
-
d_target
```

Positive means predicted displacement belongs more strongly to requested goal's local transition family.

This is a new diagnostic.

Do not use it as a classification loss.

---

# 27. Primary Claim C13

Define:

```text
C13_language_selects_state_conditioned_transition_family
```

SUPPORTED only if ALL held-out gates pass.

## G1 — Causal language redirection preserved

```text
RedirectGain > 0
lower95 > 0

Execution RedirectGain > 0
lower95 > 0
```

and both retain at least 90% of Wave21 B1 magnitude.

## G2 — State-conditioned displacement improves future prediction

LCT-TD beats goal+horizon mean displacement on BOTH:

```text
H2 full MSE
H4 decoded action MSE
```

with paired source-session clustered CI.

## G3 — State matters beyond language

LCT-TD beats:

```text
language prototype
static goal core
goal+horizon mean displacement
```

on both required metrics.

## G4 — Horizon-specific structure matters

Horizon-conditioned support beats horizon-agnostic support on held-out endpoint prediction.

## G5 — Transition-family identity

Require:

```text
family_margin > 0
lower95 > 0
```

in full and execution displacement spaces.

## G6 — Endpoint identity

Require:

```text
endpoint macro accuracy >= 0.60
```

## G7 — Decode/reencode identity

Require:

```text
decoded/reencoded macro accuracy >= 0.60
```

## G8 — Continuity

LCT-TD continuity must be:

```text
no worse than Wave21 B1
and better than static prototype replacement
```

## G9 — Breadth

Positive family margin on at least:

```text
5/6 goals
```

and at:

```text
H1
H2
H4
```

for the aggregate.

---

# 28. Stronger Claim C14

Define:

```text
C14_language_as_state_horizon_conditioned_executable_transition_selector
```

SUPPORTED only if C13 passes and:

1. same-state six-way family identity >=0.65;
2. execution family identity >=0.65;
3. endpoint identity >=0.60;
4. decode/reencode >=0.60;
5. current-state dependence passes;
6. continuity passes;
7. no single goal contributes >40% of RedirectGain;
8. `lift_blue_block_slider -> place_in_slider` passes displacement cosine, family margin, and decoded MSE.

Safe wording:

> **Language selects a state- and horizon-conditioned family of executable latent displacements, while the current latent determines the specific trajectory realized within that family.**

---

# 29. Canonical Lift-to-Place Case

For all eligible held-out:

```text
lift_blue_block_slider -> place_in_slider
```

report at:

```text
H1
H2
H4
```

for:

```text
Wave21 B1
goal prototype
static goal core
goal+horizon mean displacement
source-conditioned D2
LCT-TD if authorized
ground truth
```

Metrics:

```text
displacement cosine
execution displacement cosine
norm ratio
future latent MSE
decoded action MSE
family margin
endpoint identity
continuity
```

No cherry-picking.

---

# 30. Horizon-Specific Visualization

Fit PCA on TRAIN displacement vectors only.

For each goal plot:

```text
H1 displacement family
H2 displacement family
H4 displacement family
```

Overlay development/held-out predicted displacements.

Purpose:

visually test whether:

```text
one goal = one endpoint cluster
```

is weaker than:

```text
one goal = multiple horizon-dependent displacement families
```

Visualization only.

---

# 31. Source-State Conditioning Visualization

For one fixed goal:

1. choose several distinct current states;
2. show their nearest train source states;
3. show paired train displacements;
4. show predicted displacement;
5. show ground truth.

This figure should demonstrate:

```text
same language
different current state
different valid transition direction
```

---

# 32. Required Statistical Protocol

Independent unit:

```text
continuous source session
```

Use:

```text
10,000 bootstrap replicates
cluster = source session
seed = 240824
```

Same-state interventions paired within boundary.

Use goal/horizon stratification where appropriate.

---

# 33. Regression / Explanatory Analysis

On development and frozen held-out analysis, compare predictors of true future displacement/endpoint:

```text
static goal-core distance
horizon-specific core distance
source-state distance
paired displacement cosine
family margin
global cycle residual
```

Question:

> Does paired, state-conditioned transition geometry explain future dynamics beyond static endpoint geometry?

Report:

```text
standardized coefficients
incremental R^2
rank correlations
clustered CI
```

Do not claim formal causality from regression.

---

# 34. Sparse Annotation Disclosure

Preserve exactly:

Official CALVIN annotations are sparse intervals.

The next annotation's true start frame is used as the transition boundary.

All action chunks are physically contiguous in the original session.

Unannotated gaps are retained.

Do not label gap frames as the previous or next atomic action without evidence.

---

# 35. No Closed-Loop Yet

Do NOT run closed-loop robot execution in Wave 24.

Reason:

The current scientific question is whether transition-displacement structure is valid.

Closed-loop should only follow if C13/C14 are supported.

---

# 36. Required Figures

## Figure 1
Static endpoint core vs horizon-specific core vs transition-displacement family.

## Figure 2
H1/H2/H4 displacement distributions per goal.

## Figure 3
Source-state proximity vs displacement similarity.

## Figure 4
D2 paired displacement prediction vs goal+horizon mean displacement.

## Figure 5
Same-state six-way language intervention.

## Figure 6
Current-state-conditioned transitions under same goal.

## Figure 7
Lift -> place H1/H2/H4.

---

# 37. Required Tables

### Table A
Paired transition inventory by goal/horizon.

### Table B
S1 vs S2 vs S3 diagnostic comparison.

### Table C
Displacement direction metrics.

### Table D
Development mechanism gate M2.

### Table E
If authorized, LCT-TD held-out results.

### Table F
C13/C14 gate table.

---

# 38. Failure Taxonomy

Use fixed categories:

```text
no horizon dependence
no source-state dependence
goal-only displacement sufficient
poor displacement direction
wrong displacement magnitude
semantic-only transition family
execution-family overlap
endpoint identity failure
decode/reencode failure
continuity failure
sparse-data cell
long-horizon accumulation
other
```

---

# 39. Required Tests

At minimum:

```text
Wave21 representation hashes unchanged
decoder hashes unchanged
text encoder unchanged
Wave21 split unchanged

paired records use original contiguous source frames
no reset crossed
no synthetic gap labels

delta_h exactly equals z_future_h - z_current

goal/horizon filtering correct
source neighbor retrieval uses source current latent
NOT endpoint latent

K = 20
tau computed TRAIN only

static core uses train only
horizon core uses train only
transition family uses train only

development-only mechanism gate
held-out unopened before M2 authorization

no endpoint attraction loss
no cycle loss
no classification loss
no prototype loss
no F2
no DEL

same-state language swap changes only language

bootstrap cluster = source session
replicates = 10000
seed = 240824

all outputs finite
all JSON valid
```

Target:

```text
all tests pass
```

---

# 40. Stop Conditions

STOP if:

```text
paired source-current / future endpoint mapping cannot be reconstructed reliably

fewer than 4 goals have adequate paired H1/H2/H4 support

M2 mechanism gate fails

source-conditioned displacement does not beat goal+horizon mean displacement

current-state dependence fails

no lambda_TM passes development selection

held-out test opened before preregistration freeze
```

If C13 fails:

do not add another rescue loss in Wave24.

---

# 41. Required Deliverables

Produce:

```text
twenty_fourth_wave_results.md
twenty_fourth_wave_next_experiment.md

wave24_frozen_manifest.json
wave24_split_freeze.json

wave24_paired_transition_inventory.parquet
wave24_paired_transition_inventory_report.md

wave24_static_core_manifest.json
wave24_horizon_core_manifest.json
wave24_transition_family_manifest.json

wave24_phaseA_horizon_core_diagnosis.md
wave24_phaseA_source_conditioned_displacement.md
wave24_mechanism_gate.json

wave24_model_preregistration.json
wave24_seed_preregistration.json
wave24_transition_weight_selection.json
wave24_final_test_preregistration.json

wave24_training_report.md
wave24_statistical_report.md

wave24_main_comparison.md
wave24_same_state_language_swap.md
wave24_transition_family_results.md
wave24_decode_reencode_results.md
wave24_continuity_results.md
wave24_lift_to_place_case.md

wave24_failure_taxonomy.md
wave24_claim_decision.json

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

# 42. Claim Decision JSON

Write:

`wave24_claim_decision.json`

with:

```text
M2_state_horizon_conditioned_displacement_family:
SUPPORTED_FOR_INTERVENTION / REJECTED

C13_language_selects_state_conditioned_transition_family:
SUPPORTED / REJECTED / NOT_TESTED

C14_language_as_state_horizon_conditioned_executable_transition_selector:
SUPPORTED / REJECTED / NOT_TESTED

horizon_specific_support_better_than_static:
true / false / inconclusive

source_state_conditioning_matters:
true / false / inconclusive

paired_displacement_predicts_future:
true / false / inconclusive

full_redirect_preserved:
true / false / inconclusive

execution_redirect_preserved:
true / false / inconclusive

endpoint_identity_repaired:
true / false / inconclusive

decode_reencode_identity_repaired:
true / false / inconclusive

continuity_preserved:
true / false / inconclusive
```

---

# 43. Final Report Questions

The final report must answer:

1. How many paired H1/H2/H4 transitions were reconstructed?
2. How many distinct source sessions?
3. Which goal/horizon cells have adequate support?
4. Does horizon-specific core beat static goal core?
5. Is HorizonCoreGain positive with lower95 > 0?
6. Do source-conditioned train displacements predict development true displacement direction?
7. What is full displacement cosine?
8. What is execution displacement cosine?
9. Does D2 beat goal+horizon mean displacement on full MSE?
10. Does D2 beat it on execution MSE?
11. Does current state therefore matter beyond goal+horizon?
12. Does D2 beat static endpoint baselines?
13. Does D2 preserve endpoint identity?
14. Does D2 preserve continuity?
15. Is M2 supported?
16. If M2 passes, which lambda_TM is selected?
17. Does LCT-TD preserve full RedirectGain?
18. Does it preserve execution RedirectGain?
19. Does LCT-TD beat goal+horizon mean displacement?
20. Does it beat language prototype?
21. Does family_margin have lower95 > 0?
22. Does endpoint macro accuracy reach >=0.60?
23. Does decode/reencode accuracy reach >=0.60?
24. Does the effect hold at H1/H2/H4?
25. Does it hold across at least 5/6 goals?
26. Does lift_blue_block_slider -> place_in_slider pass?
27. Is C13 supported?
28. Is C14 supported?
29. What exact mechanism now best explains Waves21–23?
30. What exact paper claim is defensible?
31. If C13 passes, what closed-loop experiment should follow?
32. If C13 fails, what remains supported from the language-vector-field hypothesis?

---

# 44. Interpretation Rules

If M2 and C13/C14 pass:

> **Language does not specify a static executable endpoint. Instead, it selects a state- and horizon-conditioned family of executable transition displacements. The current latent determines which displacement within that family is appropriate, producing a continuous language-directed trajectory.**

Chinese:

> **语言并不直接指定一个固定的可执行终点区域。它更像是在当前状态下选择一组与时间尺度相关的可执行状态转移方向，而当前 latent 决定实际采用其中哪一条 transition。**

This would refine the paper's core story to:

```text
language changes the vector field
+
current state determines local transition
+
horizon determines displacement scale
```

If horizon conditioning helps but current-state conditioning does not:

> Language selects a horizon-dependent transition family, but state-dependent correspondence is not established.

If current-state-conditioned displacement is predictive but learned LCT-TD fails:

> The transition-family geometry is explanatory, but the current regularized transition model does not yet exploit it reliably.

If M2 fails:

> Static endpoint geometry was not the only problem; the available latent representation does not exhibit a sufficiently coherent state-conditioned displacement family for the proposed mechanism.

---

# 45. Strategic Meaning

The project has now moved through three increasingly precise hypotheses:

```text
Static target endpoint
    ↓ rejected as too simple

Global executable manifold
    ↓ rejected as target-misaligned

Goal-specific executable core
    ↓ explanatory but not corrective

State- and horizon-conditioned transition family
    ↓ Wave24
```

The strongest surviving empirical fact is still:

```text
changing only language changes future latent direction
```

Wave24 asks what structure that direction actually belongs to.

The new conceptual model is:

```text
current latent z_t
        +
language goal g
        +
horizon h
        ↓
select transition family D_{g,h}(z_t)
        ↓
choose state-compatible displacement
        ↓
z_future = z_t + delta
        ↓
decode to continuous robot action
```

This formulation is directly aligned with the data observed in Waves 21–23.

That is the entire purpose of Wave 24.
