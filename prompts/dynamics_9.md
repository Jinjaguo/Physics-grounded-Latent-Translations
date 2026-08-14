# Wave 21 Codex Prompt
# Language-Conditioned Latent Transition:
# Can Language Causally Redirect Continuous Action-Latent Dynamics?

## 0. New Scientific Direction

The project direction has changed.

Do NOT continue Wave-20's rejected LIBERO refinement-family rescue as the primary goal.
Do NOT run the previously proposed tangent-projection Wave-21 experiment.
Do NOT open the untouched LIBERO final test for the old F1/F2 claim.
Do NOT reopen DEL.

The new scientific question is:

> **Given the same current action state, does changing only the next language goal causally redirect the future latent trajectory toward the corresponding action region?**

Formal version:

Given fixed `z_t`, with two different next goals `l_next^(A)` and `l_next^(B)`, does `T_theta(z_t, l_next^(A))` generate a future latent trajectory that moves toward action region A, while `T_theta(z_t, l_next^(B))` from the exact same current latent moves toward region B?

The new paper-level hypothesis is:

> **Language is not only a label attached to an action latent. Language can act as a target coordinate that changes the direction of continuous latent dynamics.**

The desired minimal example is:

```text
current action state:
robot is finishing lift_blue_block_slider

next language goal A:
place_in_slider

next language goal B:
turn_on_lightbulb

same current latent z_t
same history
same model
same weights

only next-language input changes
```

The model should produce different future latent trajectories whose endpoints move toward the corresponding language-grounded action regions.

The goal is NOT:

```text
one long instruction -> autonomous high-level planning of an entire task
```

The goal IS:

```text
current action state + externally supplied next atomic language goal
    ->
continuous latent transition
    ->
decoded continuous robot action
```

---

## 1. Preserve Historical Scientific Record

Do not rewrite previous results.

Current frozen record:

```text
CALVIN representation:
SUPPORTED semantic addressability
SUPPORTED continuous action decodability

CALVIN local refinement:
SUPPORTED

CALVIN public H1/H2 replication:
SUPPORTED

CALVIN continuous-play H1/H2/H4/H8 refinement:
SUPPORTED

CALVIN full-latent DEL:
REJECTED

CALVIN execution/decoder-grounded DEL:
REJECTED

LIBERO independent representation:
SUPPORTED after Wave 20

LIBERO old unconditional F2 long-horizon gate:
REJECTED / NOT AUTHORIZED FOR FINAL TEST
```

Wave 20 specifically showed that the independent LIBERO representation passed strongly:

```text
A2T delta = 0.91
T2A delta = 0.916667
motor ratio = 1.129209867 <= 1.15
gripper drop = 0.000441054
```

but the old unconditional F2 dynamics gate failed O1/O3/O5/O8.

This motivates changing the scientific question.

Do NOT describe Wave 20 as "language-conditioned dynamics failed".
That experiment did not test the new hypothesis.

---

## 2. Primary Domain for Wave 21

Primary domain:

```text
CALVIN continuous play
```

Reason:

The CALVIN representation already contains explicit atomic language anchors such as:

```text
lift_blue_block_slider
lift_red_block_table
place_in_slider
push_pink_block_right
turn_off_lightbulb
turn_on_lightbulb
```

These atomic labels are exactly suited to the new hypothesis.

The primary experiment must use physically continuous CALVIN play streams.
Do not concatenate disconnected demonstrations.
Do not reset latent dynamics at language annotation boundaries.

Language boundaries are target-change events.
They are NOT physical resets.

---

## 3. Frozen Representation

Use the final frozen CALVIN representation that already passed the prospective representation readiness protocol.

Record SHA256 for:

```text
action encoder
decoder
semantic projection
text encoder / text embedding model
normalization statistics
```

Write:

`wave21_frozen_representation_manifest.json`

Required:

```text
representation optimizer steps = 0
decoder optimizer steps = 0
text encoder optimizer steps = 0
EMA updates = 0
```

The representation must remain frozen for the entire experiment.

Do not reuse the LIBERO Wave-20 representation for the primary CALVIN experiment.

---

## 4. Atomic Action Vocabulary

Primary vocabulary is the six established CALVIN atomic actions:

```text
A0 = lift_blue_block_slider
A1 = lift_red_block_table
A2 = place_in_slider
A3 = push_pink_block_right
A4 = turn_off_lightbulb
A5 = turn_on_lightbulb
```

Before model training, audit all available continuous play annotations.

For every annotation boundary, record:

```text
source session
boundary frame
previous atomic label
next atomic label
frames available before boundary
frames available after boundary
reset/discontinuity status
```

Write:

`wave21_transition_inventory.csv`

and:

`wave21_transition_inventory_report.md`

---

## 5. Boundary-Aligned Continuous Latent Construction

Let an annotation change at source frame `b`.

Construct exact physically contiguous H=16 chunks:

```text
current chunk:
a[b-16:b]

future chunk 1:
a[b:b+16]

future chunk 2:
a[b+16:b+32]

future chunk 3:
a[b+32:b+48]

future chunk 4:
a[b+48:b+64]
```

Encode with the frozen action encoder:

```text
z_0 = E(a[b-16:b])
z_1 = E(a[b:b+16])
z_2 = E(a[b+16:b+32])
...
```

The chunks must be contiguous in the original source stream.

Do not insert a gap at the language boundary.
Do not construct a new simulator reset.
Do not use annotation boundaries as latent resets.

Primary minimum future support:

```text
H1 = 1 latent step = 16 frames
H2 = 2 latent steps = 32 frames
H4 = 4 latent steps = 64 frames
```

Use H8 only if a boundary has >=128 future frames and enough samples exist prospectively.

Do not require H8 for the primary gate.

This experiment is about **goal-conditioned redirection**, not maximal horizon.

---

## 6. Transition Dataset Unit

Each real observed transition sample is:

```text
(
    source_session,
    boundary_frame,
    z_prev_optional,
    z_current,
    previous_language,
    next_language,
    future_latents,
    future_actions
)
```

The primary target is the observed next-language trajectory.

Example:

```text
previous_language = lift_blue_block_slider
next_language = place_in_slider

input:
z_current + "place_in_slider"

target:
z_1, z_2, ..., z_H
```

No future action may be used as model input.

Future latents/actions are target/reference only.

---

## 7. Source-Session Split

Split by complete continuous source session.

No transition from the same source session may cross train/dev/test.

Preferred:

```text
60% train sessions
20% development sessions
20% held-out test sessions
```

If existing official/public splits already provide a stronger source-session separation, preserve them.

Write:

`wave21_session_split_manifest.json`

Freeze before model training.

---

## 8. Data Adequacy Gate

Before training, audit transition coverage.

For each next-goal label, report:

```text
number of train boundaries
number of dev boundaries
number of test boundaries
number of distinct source sessions
previous-label diversity
```

Preferred primary requirement:

```text
>= 50 training transitions per next goal
>= 15 development transitions per next goal
>= 15 held-out test transitions per next goal
```

Minimum admissible requirement:

```text
>= 25 train
>= 8 dev
>= 8 test
per next goal
```

If the six-task vocabulary cannot satisfy the minimum:

1. do not silently duplicate boundaries;
2. do not split windows from one boundary as independent transitions;
3. reduce the confirmatory vocabulary prospectively to the largest subset with adequate coverage;
4. require at least 4 distinct next-goal classes;
5. write the reduced vocabulary before training.

If fewer than 4 goals satisfy minimum coverage:

STOP.

Write:

`wave21_insufficient_transition_coverage.md`

---

## 9. Action-Region Definition

The experiment needs an independently defined latent region for each atomic action.

Do NOT define target regions using the Wave-21 predictor outputs.

Use only frozen representation latents from the training split.

For each atomic action `g`:

```text
R_g = set of frozen latent chunks labeled with action g
```

Define three frozen region metrics.

### 9.1 kNN distance to target region

For predicted latent `z`:

```text
d_knn(z, R_g)
=
mean distance to K nearest train latents from action g
```

Freeze:

```text
K = 20
```

before inference.

### 9.2 Target-vs-other margin

```text
margin_g(z)
=
min_{h != g} d_knn(z, R_h)
-
d_knn(z, R_g)
```

Positive means the prediction is closer to the requested target region than to any competing action region.

### 9.3 Frozen semantic retrieval score

Use the existing frozen semantic projection/text embeddings.

For each predicted latent:

```text
retrieve among six atomic language labels
```

Report:

```text
target top-1
target rank
target cosine margin
```

No Wave-21 predictor output may be used to fit these region definitions.

Write:

`wave21_action_region_manifest.json`

---

## 10. New Language-Conditioned Transition Model

Train a new dynamics model specifically for this hypothesis.

Call it:

```text
LCT = Language-Conditioned Transition
```

The representation remains frozen.

Primary input:

```text
z_current
+
optional z_previous
+
next_goal_language_embedding
```

Preferred structure:

```text
h_state = StateEncoder(z_previous, z_current)
h_goal = frozen projection(E_text(next_language))
delta_z = TransitionNet(h_state, h_goal)
z_next = z_current + delta_z
```

For multi-step rollout:

```text
z_hat_(t+1) = T(z_t, language_goal)

z_hat_(t+2) =
T(z_hat_(t+1), language_goal)

...
```

The next-language goal is held fixed throughout the target primitive rollout.

Do NOT feed future task annotations.
Do NOT give the model the ground-truth future action.

---

## 11. Semantic / Execution Handling

Preserve the frozen factorized representation:

```text
z = [z_sem, z_exec]
```

Primary LCT may predict the full next latent or predict execution plus a language-conditioned semantic target.

Choose exactly one implementation before final evaluation.

Preferred clean implementation:

```text
predict delta on full 32-D latent
condition explicitly on frozen next-language embedding
```

because the scientific question concerns the direction of the whole latent trajectory.

However, include an architecture audit to ensure the model cannot trivially overwrite semantic dimensions while ignoring execution.

Therefore always report separately:

```text
semantic-subspace trajectory
execution-subspace trajectory
decoded-action trajectory
```

---

## 12. Mandatory Baselines

Train/evaluate at least:

### B0 — Unconditional transition

```text
T_uncond(z_current)
```

Same architecture capacity where practical, but no next-language input.

Purpose:

Does language improve future transition prediction beyond current state/history alone?

### B1 — Correct-language LCT

```text
T(z_current, correct_next_language)
```

Primary method.

### B2 — Shuffled-language training control

Train the same model with next-language labels shuffled within the training split while preserving task frequencies.

Purpose:

Does structured language-target correspondence matter?

### B3 — Null-language intervention

At inference on B1, replace language embedding with a frozen null/zero embedding.

No retraining.

### B4 — Wrong-language swap

At inference on B1, keep the exact same `z_current` and substitute an incorrect atomic goal.

This is the core causal intervention.

Use all five wrong goals where computationally feasible.

---

## 13. Core Same-State Language Intervention

For every held-out current state `z_t`:

run the frozen B1 model six times:

```text
T(z_t, A0)
T(z_t, A1)
T(z_t, A2)
T(z_t, A3)
T(z_t, A4)
T(z_t, A5)
```

Everything must be identical except the language goal.

Same:

```text
z_previous
z_current
weights
normalization
rollout horizon
random seed
decoder
```

Only:

```text
next_language
```

changes.

This is the primary causal intervention.

Store every six-way trajectory.

---

## 14. Primary Scientific Quantity: Goal-Redirectability

For fixed `z_t` and target language `g`, define the predicted H-step endpoint:

```text
z_hat_H(g)
```

Primary target-region attraction:

```text
A_target(g)
=
d_knn(z_current, R_g)
-
d_knn(z_hat_H(g), R_g)
```

Positive means the trajectory moved toward the requested language region.

For the same state under wrong goal `h`:

```text
A_cross(g | h)
=
d_knn(z_current, R_g)
-
d_knn(z_hat_H(h), R_g)
```

Define causal redirect gain:

```text
RedirectGain(g,h)
=
A_target(g) - A_cross(g|h)
```

The primary hypothesis requires:

```text
mean RedirectGain > 0
```

under paired same-state comparisons.

This directly asks whether changing only language changes the future trajectory in the requested semantic direction.

---

## 15. Endpoint Goal Classification

For each six-way intervention endpoint:

```text
z_hat_H(g)
```

classify its nearest action region using only frozen training regions.

Metric:

```text
Goal Region Top-1 Accuracy
```

Question:

> If we ask six different next-language goals from the same current latent, does each predicted endpoint become closest to the corresponding requested action region?

Chance level with six actions:

```text
1/6
```

Also report:

```text
macro accuracy
per-goal accuracy
mean rank
target margin
```

---

## 16. Pairwise Counterfactual Direction Test

For every pair of goals `g` and `h` from the same `z_t`:

```text
Delta_(g,h)
=
z_hat_H(g) - z_hat_H(h)
```

Measure whether the difference aligns with the frozen difference between target action regions.

Define training-region prototypes only for this diagnostic:

```text
mu_g = mean train latent for action g
mu_h = mean train latent for action h
```

Compute:

```text
cos(
    z_hat_H(g) - z_hat_H(h),
    mu_g - mu_h
)
```

Positive mean cosine indicates language swaps move trajectories in the same relative direction as the corresponding action regions.

This is secondary to kNN region metrics.

---

## 17. Observed-Next-Goal Prediction Accuracy

For the real observed next annotation `g*`, compare:

```text
B0 unconditional
B1 correct-language
B2 shuffled-language
B3 null-language
```

against the actual future latent trajectory.

Metrics:

```text
H1 latent MSE
H2 latent MSE
H4 latent MSE

semantic MSE
execution MSE

decoded continuous-action MSE

execution kNN radius

target-region distance

target-region margin
```

Primary expectation:

```text
B1 correct-language < B0
B1 correct-language < B2
B1 correct-language < B3
```

for future prediction error / target-region distance.

---

## 18. Source-to-Target Transition Specificity

The model must not simply output a fixed prototype for each language.

Test this explicitly.

For the same requested next goal `g` but different current states `z_i` and `z_j`:

measure how much trajectory variation is preserved.

Required diagnostics:

```text
within-goal endpoint variance
pairwise endpoint distance
decoded-action diversity
correlation between current-state variation and predicted-trajectory variation
```

Compare against a language-only prototype baseline:

```text
z_hat = mu_g
```

and/or:

```text
trajectory = fixed mean target trajectory for language g
```

B1 must outperform a pure language-prototype model on observed future prediction.

Otherwise the system may just be retrieving a task centroid.

---

## 19. Transition-vs-Retrieval Ablation

Mandatory baseline:

```text
Language Prototype
```

Given language `g`:

```text
predict nearest/mean latent region for g
```

without using current `z_t`.

Compare to LCT.

This asks:

> Does the current latent matter, or is language alone sufficient to choose a generic action template?

Required:

```text
LCT future latent error < language-only prototype
LCT decoded action error < language-only prototype
```

on observed next-goal transitions.

---

## 20. Continuous Boundary Test

The new story explicitly requires a continuous transition rather than a discontinuous latent reset.

Measure at the boundary:

```text
jump magnitude:
||z_hat_1 - z_current||

decoded action discontinuity:
||a_hat_first - a_current_last||

execution-subspace jump

semantic-subspace jump
```

Compare:

```text
LCT
language-prototype baseline
ground-truth transition
```

The LCT trajectory should preserve transition continuity better than direct prototype replacement.

Do NOT require zero jump.

A real transition can change direction.

The claim is:

```text
continuous state-dependent transition
```

not:

```text
instant teleport to target centroid
```

---

## 21. Multi-Step Target Attraction Curve

For each target goal `g` and rollout horizon `h`:

```text
h = 0,1,2,4
```

compute:

```text
d_knn(z_hat_h, R_g)
target margin
semantic target rank
execution kNN radius
```

Plot target-region attraction over time.

Desired qualitative pattern:

```text
current latent
    ->
progressively closer to requested target region
```

Do not require monotonicity at every single step as a hard gate unless preregistered.

Primary gate should use net H4 improvement.

---

## 22. Decode-and-Reencode Consistency

For every predicted latent:

```text
z_hat
    ->
decoder
    ->
a_hat
```

Then re-encode the decoded action chunk with the frozen encoder:

```text
z_cycle = E(a_hat)
```

Measure:

```text
||z_cycle - z_hat||
```

and target-region identity of `z_cycle`.

Purpose:

Ensure language steering does not exploit latent directions that the decoder cannot realize.

Report:

```text
cycle latent error
decoded target-region top-1
decoded/reencoded target margin
```

This is important for the phrase:

```text
language redirects executable latent dynamics
```

---

## 23. Primary Gate: Language Causally Redirects Latent Dynamics

Define claim:

```text
C7_language_conditioned_transition
```

SUPPORTED only if ALL primary conditions pass on held-out source sessions.

### G1 — Correct language improves observed transition prediction

Require B1 correct-language to beat B0 unconditional on:

```text
H2 execution latent MSE
H4 decoded action MSE
```

with paired source-session-clustered 95% CI excluding zero in the favorable direction.

### G2 — Same-state language swap causes target-specific redirection

For the exact same `z_current`:

```text
mean RedirectGain > 0
```

with source-session-clustered lower 95% > 0.

### G3 — Requested target region is identifiable

Six-way Goal Region Top-1 endpoint accuracy:

```text
significantly above chance
```

and macro accuracy must exceed a frozen preregistered threshold.

Recommended threshold:

```text
>= 0.60
```

for the six-action vocabulary.

If reduced to 4–5 actions due to data adequacy, freeze a corresponding threshold before training.

### G4 — Current state matters

LCT must beat the language-only prototype baseline on observed future:

```text
H2 latent MSE
H4 decoded action MSE
```

### G5 — Executability is preserved

Decode-and-reencode cycle must remain inside a frozen acceptable range defined from ground-truth held-out action chunks.

Define the tolerance using development ground-truth reconstruction statistics before test.

### G6 — Redirection is present in execution dimensions

Execution-subspace RedirectGain must be positive with lower 95% > 0.

This prevents a trivial result where only semantic coordinates move while motor coordinates remain unchanged.

---

## 24. Stronger Claim Gate: Full Atomic-Action Transition

Define:

```text
C8_language_targeted_atomic_transition
```

SUPPORTED only if, in addition to C7:

1. H4 endpoints are closer to the requested action region than competing regions;
2. decoded/reencoded H4 endpoints preserve target identity;
3. target attraction holds across at least 4 distinct next-goal classes;
4. no single source-transition pair contributes more than 40% of the aggregate effect;
5. transition continuity is better than direct target-prototype replacement.

Safe wording if supported:

> **Given a fixed current action state, changing only the next atomic language goal causally redirects the predicted continuous latent trajectory toward the corresponding executable action region.**

Do NOT write:

> language autonomously plans arbitrary long-horizon tasks

unless separately tested.

---

## 25. Canonical `lift -> place` Case Study

Create a dedicated qualitative/quantitative case study for:

```text
current action:
lift_blue_block_slider

new next-language target:
place_in_slider
```

For each eligible held-out boundary of this transition:

show:

```text
z_current
predicted trajectory under place_in_slider
predicted trajectory under each wrong language
ground-truth place trajectory
decoded actions
target-region distance over rollout
```

Generate a 2-D visualization only as a descriptive figure:

```text
PCA or UMAP fitted on TRAIN latents only
```

Never use 2-D geometry for the statistical claim.

Figure should visually show:

```text
same starting point
different language arrows
different future latent paths
```

---

## 26. Additional Pairwise Case Studies

If data exist, include at least three distinct source/target transitions, e.g.:

```text
lift_blue_block_slider -> place_in_slider

lift_red_block_table -> place_in_slider

push_pink_block_right -> turn_on_lightbulb
```

Use the actual available transitions from the inventory.

Do not fabricate pairs absent from continuous data.

---

## 27. Language Paraphrase Robustness

Secondary experiment only.

For each atomic action, construct 3–5 frozen paraphrases before test inference.

Example:

```text
place_in_slider
put the object into the slider
move it into the slider compartment
place it inside the slider
```

Use the frozen text encoder.

Question:

> Does the target direction depend on action meaning rather than one exact annotation string?

Report:

```text
within-goal paraphrase endpoint variance
target-region accuracy
RedirectGain
```

Do not use paraphrases for training if testing semantic robustness.

---

## 28. Adversarial / Semantically Wrong Language Controls

Secondary controls:

```text
wrong valid atomic goal
empty string
unrelated sentence
shuffled language embedding
```

The strongest causal control is still:

```text
same state + different valid next atomic goal
```

Do not make nonsense prompts the primary evidence.

---

## 29. Optional Refinement Integration

Do NOT include refinement in the primary C7 gate.

Only after the language-conditioned transition claim passes on development may run a secondary comparison:

```text
LCT
vs
LCT + frozen/generic refinement
```

Question:

> Does refinement stabilize language-directed transitions without changing the selected target direction?

This is secondary.

Do not use refinement to rescue a failed language-conditioned transition claim.

Do not run DEL.

---

## 30. Statistical Unit

Highest-level independent unit:

```text
continuous source session
```

Boundaries from the same session are clustered.

Use:

```text
10,000 bootstrap replicates
cluster = source session
seed = 210821
```

Where task balance matters, use task/next-goal stratification.

Do not bootstrap windows or language swaps as independent observations.

Same-state six-way language interventions are paired within each boundary.

---

## 31. Held-Out Test Discipline

Before held-out test inference freeze:

```text
representation hashes
LCT architecture
all model seeds
training epochs
learning rate
language embedding interface
rollout horizons
action-region definitions
K=20
primary metrics
bootstrap seed
all claim gates
case-study selection rule
```

Write:

`wave21_final_test_preregistration.json`

Do not tune after opening held-out sessions.

---

## 32. Model Seeds

Use at least:

```text
6 registered seeds
```

for:

```text
B0 unconditional
B1 correct-language LCT
B2 shuffled-language control
```

Pair initializations where practical.

Do not add seeds after seeing test results.

Primary statistics may aggregate per frozen selection rule, but seed robustness must be reported.

Freeze selection rule before test.

---

## 33. Training Objective

Primary LCT training objective:

```text
L =
lambda_latent * latent_prediction_loss
+
lambda_decode * decoded_action_loss
```

Do NOT add an explicit target-region attraction loss in the primary model.

Reason:

The primary scientific test asks whether next-language conditioning naturally redirects the learned dynamics toward the correct action region.

If the model is explicitly trained to minimize region distance, the target-region result becomes partially built into the loss.

Preferred primary objective:

```text
predict observed future latent trajectory
conditioned on next language
```

with standard latent + decoded-action supervision.

Freeze loss weights on development only.

No held-out test tuning.

---

## 34. Avoid Trivial Semantic Leakage

Because `z` has semantic dimensions aligned to language, the model could pass a target-region metric by directly copying language into `z_sem`.

Therefore C7 requires execution-space evidence.

Mandatory:

```text
execution-subspace RedirectGain
decoded-action MSE
decoded/reencoded target-region accuracy
```

Additionally run:

```text
semantic-only distance
execution-only distance
full-latent distance
```

If only semantic-space steering is positive but execution-space steering fails:

```text
C7 = NOT SUPPORTED
```

Safe conclusion would then be:

> Language redirects semantic coordinates but has not been shown to redirect executable action dynamics.

---

## 35. Avoid Current-State Ignorance

The model could ignore `z_current` and output a language prototype.

Therefore mandatory diagnostics:

1. language-only prototype baseline;
2. same language, different current states;
3. endpoint diversity;
4. conditional residual analysis showing that current-state variation still explains endpoint variation after controlling for language.

At minimum require LCT to outperform language-only prototype on held-out future trajectory prediction.

---

## 36. Transition Continuity Diagnostics

For every transition calculate:

```text
latent velocity before boundary
predicted latent velocity after boundary
ground-truth latent velocity after boundary

direction-change angle
velocity magnitude ratio
decoded-action jump
```

Question:

> Does language redirect an ongoing trajectory rather than reset it?

Do not require the direction-change angle to be small.
A new atomic goal can legitimately change direction.

The important comparison is:

```text
LCT vs direct prototype replacement
```

LCT should produce a more physically/temporally continuous decoded action transition.

---

## 37. Required Figures

Generate raw CSV/JSON for all figures.

### Figure 1 — Core concept

Same current latent, six next-language goals, six predicted future trajectories.

Use train-fitted PCA/UMAP only for visualization.

### Figure 2 — Same-state causal redirection

For each target:

```text
correct-language target attraction
wrong-language target attraction
RedirectGain
```

### Figure 3 — Endpoint region classification

6x6 matrix:

```text
requested language
vs
nearest predicted endpoint action region
```

### Figure 4 — Observed future prediction

B0 vs B1 vs B2 vs B3:

```text
H1/H2/H4 execution MSE
decoded action MSE
```

### Figure 5 — Current state matters

LCT vs language-only prototype.

### Figure 6 — Executability

```text
predicted latent
decoded action
re-encoded latent
target region identity
```

### Figure 7 — Canonical lift -> place example

Show:

```text
same lift endpoint
place language trajectory
wrong-language trajectories
ground truth
```

---

## 38. Required Tables

### Table A — Transition inventory

Rows:

```text
previous action
next action
train/dev/test boundary counts
distinct sessions
```

### Table B — Main held-out metrics

Columns:

```text
B0 unconditional
B1 correct LCT
B2 shuffled-language
B3 null-language
language prototype
```

Rows:

```text
H1 exec MSE
H2 exec MSE
H4 exec MSE
H4 decoded MSE
target distance
target margin
execution kNN radius
```

### Table C — Causal intervention

Rows per requested goal:

```text
RedirectGain
Goal Region Top-1
mean target margin
execution RedirectGain
```

### Table D — Claim decisions

```text
C7 language-conditioned transition
C8 atomic-action transition
```

---

## 39. Failure Taxonomy

Classify failures into:

```text
no language sensitivity
semantic-only steering
wrong target attraction
prototype collapse
current-state ignored
execution drift
decoder mismatch
trajectory discontinuity
source-transition sparsity
class imbalance
long-horizon accumulation
other
```

Do not invent categories after observing which model wins.

---

## 40. Required Tests

At minimum:

```text
frozen representation hashes unchanged
decoder hashes unchanged
text encoder unchanged

continuous source frame indexing correct
boundary chunks physically contiguous
no reset crossed
no future action input
no future annotation input beyond supplied next goal

session split disjoint
region definitions use train only
PCA/UMAP visualization fit on train only

same-state language interventions truly share identical z_current
only language tensor differs

B0 has no language input
B1 receives correct language
B2 training labels are correctly shuffled
B3 uses frozen null language
wrong-language swaps exclude correct label

language prototype baseline ignores current state

execution/full/semantic metrics separated

decode-reencode uses frozen encoder/decoder
no target-region loss in primary LCT

bootstrap clusters by source session
10,000 bootstrap replicates
seed = 210821

all outputs finite
all JSON valid
```

Target:

```text
all tests pass
```

---

## 41. Stop Conditions

STOP and report if:

```text
fewer than 4 next-goal classes meet minimum coverage

source-session split leakage exists

boundary chunks cross resets/discontinuities

representation or decoder hashes change

future target actions enter model inputs

held-out test is opened before preregistration freeze

primary LCT uses an explicit region-attraction loss

same-state language swap changes any input other than language

execution-space RedirectGain cannot be computed

decode/reencode pipeline is inconsistent
```

Do not redesign the model after test results.

---

## 42. Required Deliverables

Produce:

```text
twenty_first_wave_results.md
twenty_first_wave_next_experiment.md

wave21_frozen_representation_manifest.json
wave21_transition_inventory.csv
wave21_transition_inventory_report.md
wave21_session_split_manifest.json
wave21_action_region_manifest.json

wave21_model_preregistration.json
wave21_seed_preregistration.json
wave21_final_test_preregistration.json

wave21_training_report.md
wave21_statistical_report.md

wave21_observed_transition_results.md
wave21_same_state_language_swap_results.md
wave21_endpoint_region_results.md
wave21_execution_redirect_results.md
wave21_decode_reencode_results.md
wave21_continuity_results.md
wave21_paraphrase_results.md

wave21_failure_taxonomy.md
wave21_claim_decision.json

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

## 43. Claim Decision JSON

Write:

`wave21_claim_decision.json`

with:

```text
C7_language_conditioned_transition:
SUPPORTED / REJECTED / NOT_TESTED

C8_language_targeted_atomic_transition:
SUPPORTED / REJECTED / NOT_TESTED

language_changes_future_direction:
true / false / inconclusive

execution_space_redirection:
true / false / inconclusive

current_state_contributes_beyond_language:
true / false / inconclusive

continuous_transition_better_than_prototype_reset:
true / false / inconclusive
```

Include every primary metric and confidence interval.

---

## 44. Final Report Questions

The final report must answer:

1. How many physically continuous annotation transitions were found?
2. How many distinct source sessions?
3. Which next-goal classes met the coverage gate?
4. How many train/dev/test transitions per goal?
5. Was the representation completely frozen?
6. Was the decoder completely frozen?
7. Did B1 correct-language beat B0 unconditional on observed future prediction?
8. Did B1 beat shuffled-language?
9. Did B1 beat null-language?
10. From the exact same current latent, did changing only language change the future trajectory?
11. Was mean RedirectGain positive?
12. Was its clustered lower 95% > 0?
13. Did the execution-subspace RedirectGain also pass?
14. What was endpoint target-region accuracy?
15. Was the result above chance and above the frozen threshold?
16. Did language-only prototype retrieval perform worse than LCT?
17. Therefore, does current latent state contribute beyond language?
18. Did decoded/reencoded trajectories preserve the requested target identity?
19. Did language redirection remain executable?
20. Was the latent transition smoother than direct prototype replacement?
21. Did `lift_blue_block_slider -> place_in_slider` work as a held-out case?
22. Did the phenomenon hold across at least four target atomic actions?
23. Did paraphrases preserve the redirection effect?
24. Is C7 supported?
25. Is C8 supported?
26. What exact paper claim is scientifically defensible?
27. What is the next experiment needed to move from latent transition to closed-loop robot execution?

---

## 45. Interpretation Rules

If C7 and C8 pass, safe central wording:

> **Given the same current action state, changing only the next atomic language goal causally redirects the predicted continuous latent trajectory toward the corresponding executable action region.**

Longer paper story:

> **Language first anchors meaningful and executable action coordinates. More importantly, language can also serve as a target coordinate for latent dynamics: from the same current action state, changing the next language goal changes the direction of future latent evolution, producing a continuous state-dependent transition toward the requested atomic action region.**

Chinese:

> **语言不仅能够标记动作 latent 所处的语义区域，还可以直接作为 latent dynamics 的下一目标坐标。在保持当前动作状态完全相同的情况下，仅改变下一原子动作的语言目标，就能够系统性地改变未来 latent trajectory 的演化方向，并将其引向对应的可执行动作区域。**

If only semantic-space steering passes:

> **Language changes semantic latent coordinates, but executable dynamics redirection is not yet established.**

If LCT behaves like a language prototype and ignores current state:

> **Language selects an action region, but the model has not learned a state-dependent transition law.**

If observed future prediction improves but wrong-language interventions do not redirect endpoints:

> **Language helps prediction, but causal target-coordinate behavior is not supported.**

Do not overclaim.

---

## 46. Strategic Meaning

This wave changes the role of language from:

```text
language = descriptor / annotation of latent
```

to:

```text
language = control variable / next target coordinate
```

The scientific progression becomes:

```text
1. action chunks form continuous latent coordinates

2. language identifies meaningful atomic-action regions

3. latent trajectories are predictable

4. language supplied at the current state changes
   the direction of future latent evolution

5. the resulting trajectory can be decoded back
   into continuous robot actions
```

This is a stronger and cleaner thesis than asking the model to autonomously decompose a long natural-language instruction into an entire task plan.

The model is only asked to solve:

```text
I am here now.
The next atomic goal is there.
Generate the continuous latent transition from here to there.
```

That is the core Wave-21 experiment.
