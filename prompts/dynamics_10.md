# Wave 22 Codex Prompt
# Executable Coordinate Consistency for Language-Conditioned Latent Dynamics
# Why Does Language Redirect the Vector Field but Leave Decoder-Consistent Coordinates?

## 0. Scientific Motivation

Wave 21 produced a highly specific result.

The new language-conditioned transition hypothesis was **not globally supported**:

```text
C7_language_conditioned_transition = REJECTED
C8_language_targeted_atomic_transition = REJECTED
```

However, several key components passed strongly:

```text
Language RedirectGain = 0.250126
95% CI = [0.136495, 0.370798]

Execution RedirectGain = 0.183855
95% CI = [0.100917, 0.263777]

Correct-language LCT beat:
- unconditional transition
- shuffled-language transition
- null-language transition

on observed future prediction metrics.
```

Thus:

> Language already changes the future latent vector field in the correct causal direction, including in the execution subspace.

The failure happened downstream:

```text
endpoint macro accuracy = 0.516260 < frozen 0.60 threshold

decode -> re-encode target identity accuracy = 0.329268

cycle error = 2.800412
development tolerance = 1.434166

LCT continuity worse than direct prototype replacement
```

Therefore the Wave-22 scientific question is:

> **Why does language-conditioned redirection leave the decoder-consistent executable coordinate set, and can executable-coordinate consistency be restored without destroying the causal language redirection effect?**

This wave is a mechanistic adjudication.

Do NOT reinterpret Wave 21 as a positive C7/C8 result.

Do NOT reopen DEL.

Do NOT use the old unconditional F2 refinement as a rescue.

Do NOT add a target-region attraction loss to force endpoints into the desired language region.

The primary objective is to isolate whether the missing ingredient is:

```text
decoder consistency / executable-coordinate geometry
```

rather than:

```text
language direction itself
```

---

## 1. Frozen Wave-21 Facts

Preserve exactly:

```text
Wave21 C7 = REJECTED
Wave21 C8 = REJECTED

RedirectGain = 0.250126
CI = [0.136495, 0.370798]

Execution RedirectGain = 0.183855
CI = [0.100917, 0.263777]

Endpoint macro accuracy = 0.516260

cycle error = 2.800412
cycle tolerance = 1.434166

decoded/reencoded target identity = 0.329268

continuity gate = FAIL
```

Wave-21 LCT must remain available as the frozen reference model.

Record SHA256 hashes for:

```text
Wave21 representation
Wave21 decoder
Wave21 encoder
Wave21 text projection
Wave21 B0 unconditional
Wave21 B1 correct-language LCT
Wave21 B2 shuffled-language
normalization statistics
```

Write:

`wave22_frozen_wave21_manifest.json`

---

## 2. Primary Hypothesis

The primary Wave-22 hypothesis is:

> **Language-conditioned LCT predicts a useful direction, but its rollout drifts off the latent subset that is self-consistent under the frozen encoder-decoder pair.**

Define the frozen cycle map:

```text
C(z) = E(D(z))
```

where:

```text
E = frozen action encoder
D = frozen action decoder
```

For a decoder-consistent executable coordinate:

```text
C(z) ≈ z
```

Define cycle residual:

```text
r_cycle(z) = ||E(D(z)) - z||
```

The core hypothesis is:

```text
Wave21 LCT correction direction is language-useful,
but repeated rollout increases r_cycle,
and increased r_cycle predicts endpoint/action failure.
```

Wave 22 must test this before training a new model.

---

## 3. Phase A — Pure Diagnosis on Frozen Wave-21 Models

Before any optimizer step, analyze the frozen Wave-21 held-out trajectories.

Do NOT modify B1.

For every held-out same-state language intervention and observed transition, log at each rollout step:

```text
z_h
D(z_h)
E(D(z_h))
cycle residual
semantic cycle residual
execution cycle residual
target-region distance
execution target-region distance
endpoint target rank
decoded action MSE
execution kNN radius
local-PCA normal distance
trajectory jump
```

Evaluate for:

```text
h = 0,1,2,3,4
```

---

## 4. Cycle Residual Decomposition

Because the representation is factorized:

```text
z = [z_sem, z_exec]
```

compute:

```text
r_cycle_full
r_cycle_sem
r_cycle_exec
```

Primary question:

> Is the large cycle error mainly caused by semantic dimensions, execution dimensions, or both?

Report:

```text
mean
median
P90
P95
per-target-action
per-source-action
per-transition-pair
per-rollout-step
```

---

## 5. Does Language Redirection Point Off the Decoder-Consistent Set?

For each LCT step define:

```text
delta_lang = z_next_LCT - z_current
```

Define local decoder-consistency correction:

```text
delta_cycle = E(D(z_next_LCT)) - z_next_LCT
```

Evaluate:

```text
cos(delta_lang, delta_cycle)
```

and separately in execution dimensions.

Interpretation:

- strongly negative cosine may indicate language steering moves away from cycle-consistent coordinates;
- near zero may indicate orthogonal geometry mismatch;
- positive may indicate a different failure mechanism.

Do not pre-commit to the sign.

---

## 6. Cycle Residual vs Behavioral / Geometric Error

On frozen Wave-21 predictions, test association between:

```text
cycle residual
```

and:

```text
future latent error
decoded action error
target-region distance
target-region classification error
execution kNN radius
local-PCA normal distance
continuity jump
```

Use source-session clustered statistics.

Report:

```text
Spearman rho
Pearson r
bootstrap 95% CI
```

Primary mechanistic prerequisite:

```text
higher cycle residual should be positively associated
with worse decoded/action/geometric outcomes
```

If there is no such association, do NOT assume decoder consistency is the correct mechanism.

---

## 7. Fixed-Point / Projection Diagnostic

The frozen cycle map itself can be used diagnostically.

Given a Wave-21 LCT prediction:

```text
z^(0) = z_LCT
z^(1) = E(D(z^(0)))
z^(2) = E(D(z^(1)))
...
```

Run exactly:

```text
K_cycle = 4
```

iterations.

No training.

Call this:

```text
CYCLE_FIXED_POINT_DIAGNOSTIC
```

Measure iteration 0→4:

```text
cycle residual
target-region distance
execution target-region distance
decoded action error
endpoint target accuracy
execution kNN radius
continuity
```

Important:

This is diagnostic only.

Do not yet claim this is the final method.

Question:

> Does repeatedly applying the frozen encoder-decoder map naturally pull LCT predictions toward a decoder-supported subset, and if so, does that preserve or destroy the language-selected direction?

---

## 8. Direction Preservation Under Frozen Cycle Projection

For each same-state language swap:

```text
z_LCT(g)
```

and projected version:

```text
z_cycle4(g)
```

compute:

```text
RedirectGain_LCT
RedirectGain_cycle4

ExecutionRedirectGain_LCT
ExecutionRedirectGain_cycle4
```

Also compute:

```text
cos(
  z_cycle4(g) - z_current,
  z_LCT(g) - z_current
)
```

This is critical.

A useful executable-coordinate correction should:

```text
reduce cycle residual
while retaining target-specific language redirection
```

If cycle projection destroys RedirectGain, then a naive projection is not sufficient.

---

## 9. Phase-A Decision Gate

Before training any new model, adjudicate whether decoder consistency is a plausible mechanism.

Define:

```text
M0_decoder_consistency_mechanism
```

SUPPORTED_FOR_INTERVENTION only if all of the following hold:

```text
A1:
cycle residual increases materially over LCT rollout
relative to held-out ground-truth future latents

A2:
execution cycle residual is nontrivial

A3:
cycle residual is positively associated with
decoded-action error and/or endpoint failure

A4:
frozen cycle projection reduces cycle residual

A5:
cycle projection does not eliminate language RedirectGain
```

If A1-A5 collectively fail:

STOP.

Do not train a cycle-consistency model.

Write:

`wave22_decoder_consistency_mechanism_rejected.md`

The next wave should investigate a different mechanism.

---

## 10. Primary New Model: Cycle-Consistent Language-Conditioned Transition

Only if Phase A authorizes intervention.

Call:

```text
LCT-CC
```

Start from the exact Wave-21 B1 architecture.

No representation updates.

No decoder updates.

No encoder updates.

No text encoder updates.

Train only the transition model.

The new training objective is:

```text
L_total =
L_latent_prediction
+
lambda_decode * L_decoded_action
+
lambda_cycle * L_cycle
```

where:

```text
L_cycle =
|| E(D(z_pred)) - z_pred ||^2
```

Do NOT include:

```text
target-region attraction loss
prototype loss
language-classification endpoint loss
kNN loss
local-PCA normal-distance loss
```

The point is to test decoder consistency, not force the answer.

---

## 11. Single-Factor Design

Wave 22 must be a single-mechanism comparison.

Train exactly:

```text
B1 = frozen/reference Wave21 LCT
C1 = LCT-CC
```

plus frozen evaluation controls:

```text
B0 = Wave21 unconditional
B2 = Wave21 shuffled-language
B3 = null-language inference
P = language prototype
```

Do not redesign architecture.

The only new scientific factor is:

```text
cycle-consistency regularization
```

---

## 12. Choosing lambda_cycle

Do not sweep on the held-out test.

Use development only.

Allowed preregistered candidate set:

```text
lambda_cycle ∈ {0.1, 0.3, 1.0}
```

This is the only allowed small development sweep.

Selection rule must be frozen before evaluation:

Choose the smallest lambda satisfying on development:

```text
cycle error <= Wave21 development tolerance
AND
RedirectGain >= 90% of Wave21 B1 RedirectGain
AND
Execution RedirectGain >= 90% of Wave21 B1 execution RedirectGain
```

If multiple satisfy, choose the smallest lambda.

If none satisfy:

STOP.

Do not invent another lambda.

Write:

`wave22_cycle_weight_selection.json`

---

## 13. Seeds

Use exactly six preregistered seeds for LCT-CC.

Pair with Wave-21 B1 seeds where initialization compatibility allows.

Do not add seeds after seeing results.

Write:

`wave22_seed_preregistration.json`

---

## 14. Training Data

Use the exact same Wave-21 train sessions and transition inventory.

Do not alter:

```text
train/dev/test session split
atomic action vocabulary
boundary definition
H=16 chunking
annotation-gap handling
```

Do not add synthetic transitions.

Do not fill annotation gaps with guessed labels.

---

## 15. Sparse-Annotation Limitation

Preserve Wave-21 disclosure:

Official CALVIN annotations are sparse intervals.

The frozen transition boundary is:

```text
next annotation true start frame
```

All chunks must remain contiguous in the original physical session.

Do not claim that the annotation gap represents an explicitly labeled intermediate action.

Wave 22 is about latent executability, not dense task decomposition.

---

## 16. Held-Out Primary Comparisons

On the unchanged held-out source sessions compare:

```text
Wave21 LCT
vs
LCT-CC
```

for:

```text
RedirectGain
Execution RedirectGain
endpoint macro accuracy
decoded/reencoded target accuracy
cycle error
H2 execution MSE
H4 decoded MSE
execution kNN radius
local-PCA normal distance
continuity error
```

---

## 17. Primary Claim C9

Define:

```text
C9_executable_language_redirect
```

Question:

> Can language-conditioned redirection be retained while the trajectory is restored to decoder-consistent executable coordinates?

C9 is SUPPORTED only if ALL pass.

### G1 — Language causal effect preserved

Require:

```text
RedirectGain > 0
clustered lower95 > 0
```

and

```text
LCT-CC RedirectGain >= 0.90 * Wave21 LCT RedirectGain
```

### G2 — Execution redirection preserved

Require:

```text
Execution RedirectGain > 0
clustered lower95 > 0
```

and

```text
LCT-CC execution RedirectGain >= 0.90 * Wave21 LCT execution RedirectGain
```

### G3 — Cycle consistency repaired

Require:

```text
held-out cycle error <= 1.434166
```

using the exact Wave-21 development-frozen tolerance.

Also require:

```text
LCT-CC cycle error < Wave21 LCT cycle error
```

with clustered 95% CI excluding zero in the favorable direction.

### G4 — Endpoint target identity improves

Require:

```text
endpoint macro accuracy >= 0.60
```

OR, if this hard threshold remains unmet, do NOT call C9 supported.

Do not relax the threshold post hoc.

### G5 — Decode/re-encode target identity improves

Require:

```text
decoded/reencoded macro target accuracy >= 0.60
```

### G6 — Current state still matters

LCT-CC must beat the language-prototype baseline on BOTH:

```text
H2 full-latent MSE
H4 decoded action MSE
```

### G7 — Continuity repaired

Require LCT-CC transition continuity error:

```text
< language-prototype replacement
```

and

```text
< Wave21 LCT
```

on the preregistered continuity metric.

---

## 18. Stronger Claim C10

Define:

```text
C10_language_as_executable_target_coordinate
```

SUPPORTED only if C9 passes and additionally:

1. positive target attraction on at least 5/6 goal classes;
2. execution-space target attraction on at least 5/6;
3. decoded/reencoded target identity >=0.60;
4. no single task contributes >40% of total RedirectGain;
5. continuity is improved without collapsing endpoint diversity;
6. current-state contribution remains significant beyond language.

Safe wording if supported:

> **Language changes the latent vector field, and decoder-consistency regularization keeps the resulting trajectory within executable action coordinates, allowing the same current action state to transition continuously toward different language-specified atomic goals.**

---

## 19. Prototype-Collapse Check

Cycle consistency may accidentally make the model collapse toward decoder-supported class prototypes.

Therefore measure:

```text
within-goal endpoint variance
between-current-state endpoint variance
decoded-action diversity
current-state -> endpoint residual predictability
```

Compare:

```text
Wave21 LCT
LCT-CC
language prototype
```

LCT-CC must remain clearly more state-dependent than the prototype baseline.

---

## 20. Cycle-Loss Leakage Audit

Because encoder and decoder are frozen, ensure:

```text
gradients pass through E(D(z_pred))
only into the transition model
```

Required:

```text
encoder.requires_grad = false
decoder.requires_grad = false
representation hashes unchanged
```

Add explicit unit tests.

---

## 21. Local Geometry Diagnostic

Estimate local Jacobian of the frozen cycle map:

```text
J_C(z) = d(E(D(z))) / dz
```

on a subset of development points.

Report:

```text
singular values
spectral norm
effective rank
semantic vs execution blocks
```

Purpose:

Understand whether certain latent directions are weakly supported by the encoder-decoder pair.

This is descriptive.

Do not use Jacobian values to tune LCT-CC.

---

## 22. Language Direction vs Decoder-Supported Tangent

At each current latent, estimate the local decoder-supported tangent from train latents using the same train-only local PCA protocol already used historically.

Project Wave-21 language correction:

```text
delta_lang
```

into:

```text
tangent component
normal component
```

Report:

```text
||delta_tangent||
||delta_normal||
normal fraction
per target
```

Primary mechanistic question:

> Is the failure caused by language steering containing a large decoder-unsupported normal component?

This analysis is secondary to cycle consistency, but important for interpretation.

Do not apply tangent projection as the main method in Wave 22.

---

## 23. Canonical `lift -> place` Analysis

For held-out:

```text
lift_blue_block_slider -> place_in_slider
```

show side by side:

```text
Wave21 LCT
LCT-CC
language prototype
ground truth
```

Plot across H0-H4:

```text
target-region distance
execution target-region distance
cycle residual
decoded action error
continuity
```

Also save decoded action chunks.

---

## 24. Same-State Six-Way Intervention

Repeat the exact Wave-21 same-state experiment.

For the same current latent:

```text
goal A0
goal A1
goal A2
goal A3
goal A4
goal A5
```

Only language may change.

Compare six-way endpoint confusion matrices:

```text
Wave21 LCT
LCT-CC
decode-reencoded LCT-CC
```

This is essential to show that executable consistency did not erase language controllability.

---

## 25. Multi-Step Cycle Drift

For H0/H1/H2/H4 report:

```text
cycle residual
execution cycle residual
endpoint target accuracy
decoded/reencoded target accuracy
```

Desired evidence:

```text
Wave21:
language redirection grows but cycle drift accumulates

LCT-CC:
language redirection remains
while cycle drift is suppressed
```

---

## 26. No Refinement Rescue

Do NOT add Wave15/17 F2 refinement to LCT-CC in the primary experiment.

Do NOT add:

```text
generic refinement
tangent refinement
DEL
projection after every step
```

to rescue C9.

If C9 passes, a future wave may separately test whether refinement further stabilizes the language-directed trajectory.

---

## 27. Statistical Protocol

Highest-level independent unit:

```text
continuous source session
```

Use:

```text
10,000 bootstrap replicates
cluster = source session
seed = 220822
```

Same-state language swaps are paired within boundary.

Report task-stratified results where appropriate.

---

## 28. Development/Test Discipline

Development may be used only for:

```text
lambda_cycle selection from {0.1,0.3,1.0}
cycle-consistency tolerance verification
training sanity
```

Before held-out test:

freeze:

```text
selected lambda_cycle
model seed list
checkpoint rule
all hashes
all gates
all metrics
bootstrap seed
```

Write:

`wave22_final_test_preregistration.json`

Then evaluate held-out once.

---

## 29. Required Figures

### Figure 1 — Mechanism diagnosis

Wave21 rollout step vs:

```text
RedirectGain
cycle residual
decoded error
```

### Figure 2 — Cycle residual vs failure

Scatter / binned relation:

```text
cycle error
vs
decoded action error / endpoint correctness
```

### Figure 3 — Fixed cycle-map diagnostic

Iteration 0→4:

```text
cycle residual
RedirectGain
endpoint accuracy
```

### Figure 4 — Main comparison

```text
Wave21 LCT
LCT-CC
prototype
```

on:

```text
RedirectGain
execution RedirectGain
cycle error
endpoint accuracy
decode-reencode accuracy
continuity
```

### Figure 5 — Six-way same-state redirection

Same start state, six goals:

```text
Wave21 LCT trajectories
LCT-CC trajectories
```

### Figure 6 — lift -> place

Full canonical transition case.

---

## 30. Required Tables

### Table A — Frozen Wave21 diagnosis

Rows:

```text
cycle residual
execution cycle residual
decoded error
target distance
endpoint accuracy
```

by H1/H2/H4.

### Table B — LCT vs LCT-CC

Rows:

```text
RedirectGain
Execution RedirectGain
cycle error
endpoint accuracy
decode-reencode accuracy
H2 full MSE
H4 decoded MSE
continuity
```

### Table C — Per-goal breakdown

All six atomic targets.

### Table D — Claim gates

```text
C9
C10
```

---

## 31. Failure Taxonomy

Classify:

```text
cycle drift
semantic-only direction
execution normal drift
decoder inconsistency
prototype collapse
current-state loss
continuity failure
target identity failure
goal-specific failure
long-horizon accumulation
other
```

Do not invent categories after seeing final outcomes.

---

## 32. Required Tests

At minimum:

```text
Wave21 hashes unchanged
encoder frozen
decoder frozen
text projection frozen

L_cycle gradients only update transition model

no target-region loss
no prototype loss
no kNN loss
no PCA loss

same Wave21 train/dev/test sessions
same boundary inventory
same annotation-gap handling

same-state intervention changes only language

cycle map exactly equals E(D(z))
cycle iterations exactly 4 for diagnostic

lambda_cycle only in {0.1,0.3,1.0}
selection uses development only

held-out test unopened before freeze

bootstrap cluster = source session
bootstrap replicates = 10000
seed = 220822

all outputs finite
all JSON valid
```

Target:

```text
all tests pass
```

---

## 33. Stop Conditions

STOP if:

```text
Phase-A diagnosis does not support decoder consistency as a plausible mechanism

cycle residual is not associated with decoded/geometric failure

frozen cycle projection destroys language RedirectGain completely

no lambda in {0.1,0.3,1.0} satisfies the development preservation rule

encoder/decoder hashes change

held-out test is opened early

target-region attraction loss is introduced

C9 gate fails
```

If C9 fails:

do not add another rescue mechanism in the same wave.

Write the failure cleanly.

---

## 34. Required Deliverables

Produce:

```text
twenty_second_wave_results.md
twenty_second_wave_next_experiment.md

wave22_frozen_wave21_manifest.json
wave22_phaseA_cycle_diagnosis.md
wave22_cycle_association_report.md
wave22_cycle_projection_diagnostic.md

wave22_cycle_weight_selection.json
wave22_seed_preregistration.json
wave22_model_preregistration.json
wave22_final_test_preregistration.json

wave22_training_report.md
wave22_statistical_report.md

wave22_main_comparison.md
wave22_same_state_language_swap.md
wave22_decode_reencode_results.md
wave22_continuity_results.md
wave22_geometry_analysis.md
wave22_lift_to_place_case.md

wave22_failure_taxonomy.md
wave22_claim_decision.json

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

## 35. Claim Decision JSON

Write:

`wave22_claim_decision.json`

with:

```text
M0_decoder_consistency_mechanism:
SUPPORTED_FOR_INTERVENTION / REJECTED

C9_executable_language_redirect:
SUPPORTED / REJECTED / NOT_TESTED

C10_language_as_executable_target_coordinate:
SUPPORTED / REJECTED / NOT_TESTED

language_redirect_preserved:
true / false / inconclusive

execution_redirect_preserved:
true / false / inconclusive

cycle_consistency_repaired:
true / false / inconclusive

endpoint_identity_repaired:
true / false / inconclusive

decode_reencode_identity_repaired:
true / false / inconclusive

continuity_repaired:
true / false / inconclusive

current_state_still_matters:
true / false / inconclusive
```

---

## 36. Final Report Questions

The final report must answer:

1. Does Wave21 cycle residual increase over rollout horizon?
2. Is the failure primarily semantic-cycle or execution-cycle?
3. Is cycle residual associated with decoded-action error?
4. Is cycle residual associated with endpoint target failure?
5. Does frozen E(D(.)) iteration reduce cycle residual?
6. Does that frozen projection preserve language RedirectGain?
7. Was decoder consistency therefore authorized as the Wave22 mechanism?
8. Which lambda_cycle was selected and why?
9. Did LCT-CC preserve full RedirectGain?
10. Did it preserve execution RedirectGain?
11. Did held-out cycle error fall below 1.434166?
12. Did endpoint macro accuracy reach >=0.60?
13. Did decode/reencode endpoint accuracy reach >=0.60?
14. Did LCT-CC beat the language prototype on both required future-prediction metrics?
15. Did continuity improve relative to both Wave21 LCT and prototype?
16. Did current-state dependence remain?
17. Did the model avoid prototype collapse?
18. Did the phenomenon hold across at least five of six goals?
19. Did `lift_blue_block_slider -> place_in_slider` improve?
20. Is C9 supported?
21. Is C10 supported?
22. What exact mechanism explains Wave21's failure?
23. What exact paper claim is now defensible?
24. What is the next experiment if C9 passes?
25. What is the next experiment if C9 fails?

---

## 37. Interpretation Rules

If Phase A supports the mechanism and C9/C10 pass:

> **Language already provides a causal direction in latent space, but unconstrained language-conditioned rollout can leave decoder-supported coordinates. Enforcing frozen encoder-decoder cycle consistency preserves language redirection while keeping the trajectory within executable latent regions.**

Stronger paper wording:

> **Language acts as a target coordinate for latent dynamics: changing only the next atomic language goal redirects the future vector field. The remaining requirement is geometric compatibility with executable coordinates; decoder-consistency regularization supplies this constraint and converts semantic redirection into decoder-consistent action transitions.**

Chinese:

> **语言已经能够改变 latent dynamics 的演化方向；Wave21 的失败主要来自被语言推动后的轨迹逐渐离开 encoder-decoder 所支持的可执行坐标集合。通过约束预测 latent 在 frozen decoder→encoder cycle 下保持一致，可以在保留语言重定向作用的同时，使轨迹继续停留在可解码、可执行的 latent 区域。**

If cycle error improves but endpoint identity does not:

> Decoder consistency is necessary for executability but is not sufficient for target-coordinate selection.

If RedirectGain collapses after cycle consistency:

> The language-steering direction and decoder-supported geometry are partially misaligned; simple cycle regularization cannot jointly satisfy both.

If Phase A itself fails:

> Wave21's executable failure is not primarily explained by encoder-decoder inconsistency, so cycle consistency should not be pursued further.

---

## 38. Strategic Meaning

Wave 21 established an important partial fact:

```text
language changes the vector field
```

but did not establish:

```text
language changes the vector field
AND
the resulting trajectory remains executable
```

Wave 22 should isolate exactly that missing bridge.

The scientific progression becomes:

```text
language identifies action regions
        ↓
language causally redirects latent dynamics
        ↓
unconstrained rollout leaves decoder-consistent coordinates
        ↓
test executable-coordinate consistency as the missing constraint
```

The key question is no longer:

```text
Can language move the latent?
```

Wave 21 already says yes in a paired causal sense.

The key question is:

```text
Can language move the latent
without leaving the coordinate system that the decoder can actually execute?
```

That is the entire purpose of Wave 22.
