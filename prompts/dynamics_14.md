# Wave 26 Codex Prompt
# Multi-Branch Phase-Aware Latent Dynamics Study
# Rich Causal State × Flow Design × Retrieval × Data Scale

## 0. Mission

Wave 26 must NOT be a single-intervention rescue experiment.

Wave 25 already performed a broad model-family sweep and produced a strong structural clue:

- 66 candidates were compared.
- No model jointly passed all six development requirements.
- Held-out remained unopened.
- Phase-aware latent flow was the strongest new implementation and passed 5/6 development criteria.
- CFM-16 improved endpoint identity but degraded continuity.
- Retrieval-augmented RAT-C was also competitive.
- Discrete-mode oracle was weak.
- Continuous flow best-of-8 oracle was substantially stronger.
- Causal phase proxies improved a majority of matched model families.

Therefore Wave 26 should be a **multi-branch implementation and state-information study**.

The central question is:

> **Can richer causal state and a better-structured continuous latent flow convert the already-established language-conditioned vector-field effect into trajectories that are simultaneously accurate, target-consistent, and temporally continuous?**

Do not make Wave 26 depend on one binary gate.

Instead evaluate multiple implementation branches and produce a structured evidence matrix.

---

## 1. Preserve the Core Scientific Story

The central empirical result remains Wave 21:

> **Changing only the next-goal language causally redirects predicted latent dynamics, including execution-space coordinates.**

Frozen evidence:

```text
Full RedirectGain = 0.250126
95% CI = [0.136495, 0.370798]

Execution RedirectGain = 0.183855
95% CI = [0.100917, 0.263777]
```

Do not weaken or reinterpret this because later implementations failed.

Later waves identified implementation constraints:

```text
Wave22:
global decoder consistency does not preserve target identity.

Wave23:
static goal-core geometry is explanatory but endpoint attraction is not corrective.

Wave24:
current-state neighbors predict displacement direction,
but deterministic averaging shrinks magnitude and hurts continuity.

Wave25:
continuous flow is more promising than discrete modes,
causal phase features help,
but endpoint identity and continuity remain in tension.
```

Wave 26 is an implementation-search wave for this already-observed language-conditioned dynamics.

---

## 2. Key Wave-25 Evidence That Must Drive Wave 26

Freeze these development facts:

```text
models compared = 66
eligible = 0
held-out opened = false

best new method = Phase_flow

Phase_flow:
H2 full MSE = 0.8330
D2 = 1.2081

H4 decoded MSE = 0.0448
D2 = 0.0541

continuity = 0.2017
D2 = 0.2033

RedirectGain > 0
Execution RedirectGain > 0

endpoint identity = 0.4399
D2 endpoint identity = 0.4557
```

Also:

```text
CFM-16 endpoint identity = 0.4802
continuity = 0.2128
```

This exposes an endpoint-identity / continuity trade-off.

Discrete-mode oracle:

```text
worse than D2
```

Continuous CFM best-of-8 oracle:

```text
substantially stronger
```

The causal CFM mean closed approximately 90.7% of the H2 D2-to-oracle gap.

Therefore:

> The next experiment should focus primarily on continuous conditional flow and richer causal state, while retaining strong deterministic/retrieval/factorized controls.

Do not center Wave26 on discrete clustering.

---

## 3. Methodological Motivation

Use these papers only as implementation motivation, not as proof for our mechanism:

- CoLA-Flow / temporally coherent latent action flow, arXiv:2601.23087: motivates flow matching in a latent action space with explicit temporal coherence.
- Latent Action Guided Flow Matching (LAFM), arXiv:2606.23420: motivates state-selected source priors for fragmented / heteroscedastic action distributions instead of one global isotropic source.
- 3D FlowMatch Actor, arXiv:2508.11002: motivates targeted efficient flow architectures and low-step inference.
- BAKU, arXiv:2406.07539: motivates comparing action-head parameterizations, including multimodal heads, without changing the entire upstream representation.

Wave26 remains a latent-transition study, not an end-to-end visuomotor policy study.

---

## 4. Wave-26 Design Overview

Wave26 has FIVE parallel study axes:

```text
Axis A: richer causal state
Axis B: flow architecture / prior design
Axis C: temporal and identity-aware training objectives
Axis D: retrieval / factorized / multimodal matched controls
Axis E: data-scale condition
```

Run them in a structured development matrix.

Do not require one branch to pass every criterion before other branches can run.

The goal is comparative evidence.

Only the final held-out phase remains sealed until all development decisions are frozen.

---

## 5. Frozen Assets

Freeze and hash:

```text
CALVIN action encoder
decoder
semantic projection
text encoder
normalization

Wave21 B1
Wave21 B0

Wave24 paired transitions

Wave25 final valid candidate implementations
Wave25 corrected H4 indexing
Wave25 seed-before-reset initialization protocol
```

No updates to:

```text
representation
encoder
decoder
text encoder
```

unless a later separate wave explicitly tests representation learning.

Write:

`wave26_frozen_manifest.json`

---

## 6. Dataset Split

Preserve exact source-session independence.

Current known counts:

```text
train = 257 transitions
development = 139 transitions
held-out = 164 transitions
```

Verify programmatically.

Each transition supplies:

```text
z_previous
z_current
future H1/H2/H4 latents
paired action chunks
language goal
source session
boundary metadata
```

Held-out latent/action arrays remain unopened during development.

---

## 7. Data-Scale Axis

Wave26 must explicitly test whether 257 train transitions are a limiting factor.

### D0 — 25% train

Source-session-stratified subset of existing train.

### D1 — 50% train

Source-session-stratified subset.

### D2 — 100% original train

All 257 original train transitions.

### D3 — Expanded paired-transition train pool

Audit unused CALVIN continuous-play TRAIN-eligible sessions.

Add new paired transitions only if they are:

```text
physically contiguous
constructed using the same boundary rules
source-session disjoint from development and held-out
not previously used in dev/test
```

Target:

```text
at least +100 additional paired transitions
preferred +250 or more
```

Do not alter dev/test membership.

If no truly independent extra train sessions exist, mark D3 unavailable.

Do NOT simulate extra independent sample count by pretending multiple overlapping windows are independent.

Optional correlated augmentation may be used for training, but it must be labeled augmentation, not new independent data.

Write:

`wave26_data_scale_manifest.json`

---

## 8. Learning-Curve Experiment

For the strongest representative models, train on:

```text
D0
D1
D2
D3 if available
```

Measure:

```text
H2 full MSE
H4 decoded MSE
endpoint identity
decode/reencode identity
continuity
RedirectGain
Execution RedirectGain
```

Fit descriptive learning curves.

Interpretation:

```text
strong monotonic gains with data
=> data-limited regime

little gain with more data
=> state/model representation bottleneck

identity improves while continuity stays flat
=> target-information bottleneck partly data driven

continuity improves while identity stays flat
=> temporal-dynamics bottleneck
```

Do not claim formal scaling laws.

---

## 9. Axis A — Causal State Enrichment

Wave25 suggests phase information matters.

Test multiple causal state variants.

All features must be available at or before query time.

### S0 — Historical Wave21 state

```text
z_{t-1}
z_t
language
horizon
```

### S1 — Three-latent history

```text
z_{t-2}
z_{t-1}
z_t
language
horizon
```

### S2 — Four-latent history

```text
z_{t-3:t}
language
horizon
```

### S3 — Latent + recent action history

Use at least 3 recent H16 action chunks or their causal compact features.

Encode with a small history encoder.

### S4 — Latent history + gripper state

Use causal gripper information available in recorded actions / robot state.

Possible features:

```text
current gripper sign / width
recent gripper transitions
time since last gripper sign change
gripper command velocity
```

### S5 — Latent history + contact proxy

First audit which causal contact-related fields are actually available.

If exact contact state at query time exists, use it.

If exact contact is unavailable, construct a clearly named proxy using only causal fields, only if the source fields exist.

Do not call a heuristic proxy ground-truth contact.

### S6 — Explicit learned transition-phase state

Use a compact causal phase encoder over recent latent/action history.

Candidate encoders:

```text
GRU
1-D temporal convolution
small transformer
```

Keep parameter count small.

Output:

```text
phase_state phi_t
```

Condition the transition model on:

```text
[z_t, phi_t, language, horizon]
```

### S7 — Minimal causal proprioceptive state

If current CALVIN robot state is available causally, test a controlled variant including only a small physical state such as:

```text
current gripper width/state
TCP velocity
joint velocity norm
```

Avoid adding RGB or future scene state.

This branch tests whether a small physical phase state resolves ambiguity while preserving the latent-control story.

---

## 10. State-Ablation Matrix

Do not evaluate every model with every state variant blindly.

Use three representative models first:

```text
Phase_flow
F2-C separate-head deterministic
RAT-C retrieval residual
```

Evaluate S0–S7 on development.

Determine which state variables consistently improve:

```text
identity
continuity
prediction
```

Report matched differences.

Select up to 3 causal state configurations for the full model-family sweep.

Use Pareto analysis, not one metric.

---

## 11. Explicit Phase Diagnostics

For each transition derive causal phase descriptors:

```text
latent velocity:
z_t - z_{t-1}

latent acceleration:
z_t - 2 z_{t-1} + z_{t-2}

action velocity
gripper transition
translation velocity
rotation velocity
recent direction curvature
recent speed trend
```

Test whether these variables predict:

```text
endpoint-error residual
continuity-error residual
CFM sample variance
best-of-8 oracle gap
```

This directly tests whether richer history explains the cases where CFM endpoint identity and continuity disagree.

---

## 12. Axis B1 — Baseline Phase-Flow

Reproduce valid Wave25 Phase_flow exactly.

This is the anchor.

No silent architecture changes.

Report exact reproduction before new variants.

---

## 13. Axis B2 — History-Encoded Conditional Flow

Call:

`History-CFM`

Encode recent history using:

```text
GRU / TCN / small transformer
```

Condition the flow vector field on the resulting history state.

Compare:

```text
3-chunk history
4-chunk history
```

Keep the flow target identical.

---

## 14. Axis B3 — State-Selected Prior Flow

Call:

`Prior-CFM`

Do not always initialize displacement flow from one global Gaussian.

Build a small train-only prior family.

Candidate priors:

```text
goal-conditioned Gaussian
goal+horizon-conditioned Gaussian
retrieval-conditioned Gaussian
phase-conditioned Gaussian
small learned mixture prior
```

A causal selector chooses prior parameters from:

```text
history state
language
horizon
```

Then flow transports from that selected prior to target displacement.

Primary comparison:

```text
global Gaussian source
vs
state-selected source prior
```

---

## 15. Axis B4 — Retrieval-Initialized Flow

Call:

`R-CFM`

Retrieve K causal source transitions based on enriched history state.

Construct source initialization centered near the retrieved displacement distribution.

Variants:

```text
mean retrieved prior
single nearest retrieved prior
mixture of retrieved priors
```

Then flow refines rather than generates from scratch.

This combines Wave24's useful local direction signal with Wave25's strong continuous flow.

---

## 16. Axis B5 — Streaming / Warm-Start Flow

Call:

`Streaming-CFM`

Use recent displacement / latent velocity as source initialization:

```text
delta_source ~ N(delta_previous_scaled, sigma)
```

rather than zero/global Gaussian.

Variants:

```text
delta_previous
2-step extrapolated velocity
phase-conditioned previous displacement
```

This branch is especially relevant to future online retargeting.

The model receives current state + recent motion + new language goal and updates the local flow.

---

## 17. Axis B6 — Temporally Coherent Flow

Call:

`TC-CFM`

Predict H1/H2/H4 jointly or hierarchically.

Add explicit temporal-consistency structure.

Possible losses:

```text
velocity smoothness
acceleration regularization
multi-horizon consistency
decoded action continuity
```

Do NOT force endpoint attraction.

Prefer data-relative smoothness calibrated from ground-truth transition statistics.

---

## 18. Axis B7 — Flow with State-Dependent Noise Scale

Call:

`Hetero-CFM`

Predict state-dependent source variance / flow noise scale:

```text
sigma = sigma(history, language, horizon)
```

Compare against fixed sigma.

This tests heteroscedasticity directly.

---

## 19. Axis B8 — Multi-Path Flow

Call:

`MP-CFM`

Use a small number of learned continuous source branches:

```text
K = 2 or 3
```

Each branch has a source prior and a shared/specialized flow.

A causal gate predicts branch probabilities.

Unlike Wave25 fixed discrete modes, these are learned continuous flow branches.

Compare:

```text
hard branch
soft marginal
highest-probability branch
```

---

## 20. Axis C — Identity / Continuity Trade-Off Diagnostics

Wave25 exposed:

```text
better endpoint identity can worsen continuity
```

For every flow candidate report a 2-D Pareto plane:

```text
x = continuity error
y = endpoint identity
```

Color by:

```text
H4 decoded MSE
RedirectGain
```

Do not collapse this trade-off into one scalar during development exploration.

---

## 21. Axis C1 — Multi-Horizon Trajectory Supervision

Train jointly on:

```text
H1
H2
H4
```

from one consistent trajectory representation.

Use:

```text
L_traj =
MSE(z_H1, gt_H1)
+ MSE(z_H2, gt_H2)
+ MSE(z_H4, gt_H4)
```

with decoded trajectory supervision.

This is a direct way to improve endpoint identity while retaining intermediate continuity.

---

## 22. Axis C2 — Transition-Contrastive Objective

Do NOT use static endpoint class attraction.

For a query current-state + language pair:

positive:

```text
its true future displacement / trajectory
```

negatives:

```text
wrong-language future transitions from similar current states
```

Encourage the predicted transition to be closer to the correct transition than wrong-goal transitions.

Call:

`L_transition_contrast`

This is transition-level identity, not static endpoint classification.

---

## 23. Axis C3 — Decoder-Trajectory Supervision

Wave22 showed pure cycle projection is harmful.

But decoder-space trajectory supervision is still valid.

Use ground-truth action chunks:

```text
L_decode_traj =
MSE(D(z_pred_H1/H2/H4), true action chunks)
```

Compare:

```text
latent-only
vs
latent + decoded trajectory
```

Do not add pure E(D(z))-z cycle attraction as the primary mechanism.

---

## 24. Axis C4 — Continuity-Adaptive Weighting

Derive a TRAIN-only ground-truth continuity scale.

Penalize predicted jumps only when they exceed realistic transition statistics.

Example:

```text
Huber threshold = train P90 ground-truth boundary jump
```

Freeze threshold from TRAIN only.

Do not force all trajectories toward zero acceleration.

---

## 25. Axis C5 — Causal Multi-Sample Selection

For generative flow output multiple causal samples:

```text
N = 4 or 8
```

Do NOT use future ground truth to select a sample.

Compare causal selection rules:

```text
highest model probability
lowest predicted uncertainty
closest to retrieved transition support
best continuity score from current state
train-only learned transition score
```

This explicitly tries to close the Wave25 best-of-8 oracle gap without leakage.

---

## 26. Axis D — Matched Controls

Retain non-flow controls.

### Control 1 — F2-C separate heads

Wave25 best factorized model, with enriched state.

### Control 2 — RAT-C residual

Wave25 strong retrieval model, with enriched state.

### Control 3 — Weighted affine D4

Strong deterministic local control.

### Control 4 — B1

Historical language-conditioned LCT.

### Control 5 — language prototype

Identity-oriented, weak dynamics baseline.

### Control 6 — compact MoE / learned discrete code head

A multimodal matched control.

---

## 27. Learned VQ / Discrete Head Control

Fixed clustering was weak in Wave25, but learned discrete codes may behave differently.

Call:

`VQ-Transition`

Train a small codebook on TRAIN displacement/short-trajectory targets.

Condition code prediction on:

```text
rich causal state
language
horizon
```

Decode code + residual to displacement.

Candidate codebook sizes:

```text
K in {8,16}
```

Development only.

Use as a control:

```text
learned discrete latent mode
vs
continuous flow
```

---

## 28. Development Study Structure

Do not run the full Cartesian product blindly.

### Stage 1 — State sweep

Models:

```text
Phase_flow
F2-C
RAT-C
```

States:

```text
S0-S7
```

Select up to 3 causal state variants.

### Stage 2 — Flow family sweep

With selected state variants compare:

```text
Phase_flow
History-CFM
Prior-CFM
R-CFM
Streaming-CFM
TC-CFM
Hetero-CFM
MP-CFM
```

### Stage 3 — Objective sweep

On top 3 flow variants compare:

```text
base
+ multi-horizon
+ transition contrast
+ decoded trajectory
+ continuity-adaptive
selected combinations of at most two auxiliaries
```

Avoid huge combinatorial search.

### Stage 4 — Matched non-flow controls

Run enriched:

```text
F2-C
RAT-C
D4
VQ-Transition
compact MoE
```

### Stage 5 — Data-scale experiment

Run:

```text
top 2 flow models
+ strongest non-flow control
```

on D0/D1/D2/D3.

This provides broad implementation evidence while controlling compute.

---

## 29. Development Scorecard

Do NOT use a single all-or-nothing development gate.

For every model compute six independent evidence dimensions:

```text
PRED:
H2 full MSE improvement

DECODE:
H4 decoded MSE improvement

IDENTITY:
endpoint macro improvement

CYCLE-ID:
decode/reencode identity improvement

CONT:
continuity improvement

LANG:
RedirectGain and Execution RedirectGain preservation
```

Reference baselines:

```text
Wave21 B1
Wave24 D2
Wave25 Phase_flow
Wave25 CFM-16
```

For each dimension assign:

```text
strongly improved
improved
neutral
worse
strongly worse
```

using paired development effect sizes and uncertainty.

Produce:

`wave26_development_scorecard.csv`

---

## 30. Pareto Selection

Construct Pareto fronts over:

```text
H2 full MSE
H4 decoded MSE
endpoint identity
decode/reencode identity
continuity
RedirectGain
Execution RedirectGain
```

Do not discard a model solely because one metric is slightly worse if it occupies a meaningful Pareto frontier.

Select up to 3 final candidates from different implementation families when possible:

```text
1 best phase/history flow
1 best prior/retrieval/streaming flow
1 strongest non-flow control
```

All final candidates must preserve positive full and execution RedirectGain on development.

They do NOT need to pass the same six hard gates.

Freeze candidates before held-out.

---

## 31. Data-Limitation vs State-Limitation Decision

For each top model compare:

```text
S0 state at D2 data
rich state at D2 data
S0 state at D3 expanded data
rich state at D3 expanded data
```

if D3 exists.

Interpret:

### Case A

```text
rich state >> S0
additional data small effect
```

=> missing causal phase/state information.

### Case B

```text
additional data >> D2
state enrichment small effect
```

=> sample-limited.

### Case C

```text
both help
```

=> both information and data matter.

### Case D

```text
neither helps
```

=> model/representation bottleneck.

This is one of the primary Wave26 outcomes.

---

## 32. Held-Out Preregistration

Before opening held-out freeze:

```text
up to 3 selected models
state inputs
data condition
architecture
checkpoint
seed ensemble rule
flow steps
sampling rule
sample-selection rule
all metrics
bootstrap seed
claim matrix
```

Write:

`wave26_final_test_preregistration.json`

Then evaluate all selected models on the same held-out transitions.

No held-out winner tuning.

---

## 33. Held-Out Claim Matrix

Do not define one single Wave26 pass/fail.

Define multiple claims.

### C18 — Rich causal state matters

SUPPORTED if enriched-state matched models improve over their S0 counterparts on held-out in H2 full MSE and at least one of endpoint identity / continuity / H4 decoded MSE with consistent source-session evidence.

### C19 — Continuous conditional flow is the strongest transition family

SUPPORTED if at least one flow model lies on the held-out Pareto front and beats matched deterministic/retrieval controls on H2 full MSE and H4 decoded MSE while preserving positive language redirects.

### C20 — State-selected / history-aware flow reduces the identity-continuity trade-off

SUPPORTED if the best enriched flow improves BOTH endpoint identity and continuity relative to Wave25 Phase_flow / CFM-16 trade-off references.

### C21 — More paired transition data materially helps

SUPPORTED if D3 or the scaling curves show consistent gains beyond random subset variance.

### C22 — Language and causal state jointly shape local transition distributions

SUPPORTED if:

```text
changing language at fixed state changes predicted transition distribution
AND
changing causal history/state at fixed language changes predicted transition distribution
```

with corresponding execution-space effects.

These claims can have different outcomes.

Do not force them into one gate.

---

## 34. System-Readiness Flag

Separately define:

`READY_FOR_RETARGETING_TEST`

This is an engineering readiness flag, not the only scientific outcome.

Set true only if at least one held-out model demonstrates:

```text
RedirectGain lower95 > 0
Execution RedirectGain lower95 > 0

H2 full MSE < Wave21 B1
H4 decoded MSE <= Wave21 B1

endpoint identity >= 0.55
decode/reencode identity >= 0.50

continuity <= 1.05 * Wave21 B1 continuity
```

If false, do not run closed-loop retargeting yet.

If true, Wave27 may perform online retargeting.

---

## 35. Offline Retargeting Compatibility

For all final candidates, run an offline two-goal switch diagnostic.

Example:

```text
current state z_t
goal A = place_in_slider

predict one local step -> z_A1

switch language to goal B
predict next local step from z_A1 under B
```

Compare to continuing A.

Only language changes at the switch.

Metrics:

```text
post-switch RedirectGain
Execution RedirectGain
continuity at switch
decoded action jump
distribution shift
```

No simulator execution yet.

---

## 36. History / Return Compatibility

For selected models save:

```text
z_0
z_1
...
z_k
decoded action chunks
```

Evaluate waypoint consistency:

```text
decoder validity
stored waypoint reconstruction
local return waypoint distance
```

Do NOT attempt strict physical time reversal.

This prepares the future concept:

```text
return to a previously visited recoverable state
```

---

## 37. Contact / Gripper Analysis

If causal contact information is available, stratify by:

```text
free motion
grasp/contact
transport
release-like state
```

If only proxy exists, call it a proxy explicitly.

Compare whether richer state helps especially in contact-rich transitions.

This may explain why action-only latent state is insufficient.

---

## 38. Canonical Lift -> Place Analysis

For all eligible:

`lift_blue_block_slider -> place_in_slider`

compare:

```text
B1
D2
Wave25 Phase_flow
Wave25 CFM-16
History-CFM
Prior-CFM
R-CFM
Streaming-CFM
TC-CFM
best non-flow control
```

Report H1/H2/H4:

```text
full MSE
execution MSE
decoded MSE
endpoint identity
decode/reencode identity
continuity
RedirectGain
Execution RedirectGain
```

No cherry-picking.

---

## 39. Efficiency Metrics

Future online retargeting requires fast inference.

Record:

```text
parameter count
training time
inference latency
flow steps
samples per query
GPU memory
```

Compare especially:

```text
8-step
16-step
```

flow variants.

A model that is slightly more accurate but too slow for incremental retargeting should be marked accordingly.

---

## 40. Required Figures

1. Wave25 endpoint-identity vs continuity trade-off for all 66 candidates.
2. State-enrichment matched improvements S0–S7.
3. Flow-family comparison.
4. Wave26 development Pareto front.
5. Data scaling D0/D1/D2/D3.
6. Same-state language switch and distribution change.
7. Same language with different phase/history states.
8. Canonical lift -> place.

---

## 41. Required Tables

### Table A
Causal state variants.

### Table B
State sweep results.

### Table C
Flow-family results.

### Table D
Objective-ablation results.

### Table E
Matched non-flow controls.

### Table F
Data-scale results.

### Table G
Development Pareto finalists.

### Table H
Held-out claim matrix C18-C22.

### Table I
Efficiency.

---

## 42. Statistical Protocol

Independent unit:

`continuous source session`

Use:

```text
10,000 source-session bootstrap replicates
seed = 260826
```

Use paired comparisons on identical transitions.

For development exploration, emphasize effect sizes and bootstrap intervals; do not overinterpret many development comparisons as confirmatory hypothesis tests.

Held-out selected-model comparisons are confirmatory.

---

## 43. Required Tests

At minimum:

```text
frozen encoder/decoder/text hashes unchanged
session split unchanged

held-out arrays unopened until final freeze

all causal history features use <= query time
contact proxy audit complete
no future contact/state leakage

new train data source-session disjoint from dev/test

correct H4 index reproduced
seed-before-model-initialization verified
duplicate deterministic reproducibility test

flow source-prior implementation audited
retrieval queries use train only

same-state language switch changes only language

data-scale subsets nested/frozen

all JSON valid
all outputs finite

bootstrap cluster = source session
replicates = 10000
seed = 260826
```

---

## 44. Outcome Taxonomy

Do not reduce Wave26 to PASS/FAIL.

Assign any applicable labels:

```text
STATE_LIMITED
DATA_LIMITED
MODEL_LIMITED
REPRESENTATION_LIMITED

IDENTITY_CONTINUITY_TRADEOFF_REDUCED
IDENTITY_CONTINUITY_TRADEOFF_PERSISTS

STATE_SELECTED_PRIOR_SUPPORTED
TEMPORAL_HISTORY_SUPPORTED
RETRIEVAL_SUPPORTED
FLOW_SUPPORTED
DISCRETE_CONTROL_SUPPORTED
MIXED_EVIDENCE
```

Multiple labels may apply.

---

## 45. If No Model Reaches System Readiness

Do not shrink the language-redirection claim.

Use the comparative evidence to choose Wave27.

### If richer state clearly helps

Build a dedicated phase/contact-aware latent state model.

### If more data clearly helps

Collect / mine more paired transitions.

### If state-selected prior helps most

Develop state-selected latent prior flow more fully.

### If temporally coherent flow helps most

Develop a dedicated temporally regularized latent flow.

### If retrieval wins

Use retrieval-augmented latent dynamics with memory.

### If continuous flow oracle remains strong but causal models plateau

The current causal state is missing key information.

### If all methods plateau including richer state/data

Revisit the latent representation itself.

---

## 46. If System Readiness Is Reached

Wave27 should test:

```text
online language retargeting
interruptibility
return-to-history
```

Canonical sequence:

```text
lift object
-> language: place_in_slider
-> execute 1-2 latent steps
-> change language
-> observe retargeting
```

Separately:

```text
lift
-> begin place
-> RETURN
-> recover prior stored waypoint/state
```

Do not call return physical time reversal.

Call it:

```text
return to a previously visited recoverable state
```

---

## 47. Required Deliverables

Produce:

```text
twenty_sixth_wave_results.md
twenty_sixth_wave_next_experiment.md

wave26_frozen_manifest.json
wave26_dataset_audit.md
wave26_data_scale_manifest.json

wave26_phase_diagnostics.md
wave26_contact_proxy_audit.md

wave26_state_sweep_results.md
wave26_flow_family_results.md
wave26_objective_sweep_results.md
wave26_nonflow_control_results.md
wave26_data_scale_results.md

wave26_development_scorecard.csv
wave26_development_pareto.csv
wave26_final_candidate_selection.json

wave26_model_preregistration.json
wave26_seed_preregistration.json
wave26_final_test_preregistration.json

wave26_heldout_results.md
wave26_claim_matrix.json

wave26_same_state_language_switch.md
wave26_same_language_state_ablation.md
wave26_retargeting_compatibility.md
wave26_history_return_compatibility.md
wave26_lift_to_place_case.md

wave26_efficiency_report.md
wave26_failure_taxonomy.md
wave26_statistical_report.md

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

## 48. Claim Matrix JSON

Write:

`wave26_claim_matrix.json`

with:

```text
C18_rich_causal_state_matters
C19_continuous_flow_strongest_family
C20_enriched_flow_reduces_identity_continuity_tradeoff
C21_more_transition_data_helps
C22_language_and_state_shape_transition_distribution

READY_FOR_RETARGETING_TEST

best_state_configuration
best_flow_family
best_nonflow_control
best_data_condition

state_limitation_evidence
data_limitation_evidence
model_limitation_evidence
representation_limitation_evidence

language_redirect_preserved
execution_redirect_preserved
identity_improved
decode_reencode_improved
continuity_improved

recommended_wave27_direction
```

Each scientific claim should be:

```text
SUPPORTED
NOT_SUPPORTED
MIXED
NOT_TESTED
```

Do not force one overall binary outcome.

---

## 49. Final Report Questions

The report must answer:

1. Does three-latent history improve Phase_flow?
2. Does four-latent history help further?
3. Does recent action history help?
4. Does gripper state help?
5. Does contact/contact-proxy information help?
6. Does an explicit learned phase state help?
7. Does minimal proprioception help?
8. Which causal state variant is best?
9. Does richer state specifically reduce endpoint/continuity disagreement?
10. Does History-CFM beat Phase_flow?
11. Does state-selected Prior-CFM help?
12. Does retrieval-initialized R-CFM help?
13. Does Streaming-CFM help?
14. Does TC-CFM improve continuity?
15. Does Hetero-CFM improve identity or uncertainty calibration?
16. Does MP-CFM help?
17. Does multi-horizon supervision help?
18. Does transition-contrastive supervision help?
19. Does decoder-trajectory supervision help?
20. Does adaptive continuity loss help?
21. Can causal sample selection close part of the best-of-8 oracle gap?
22. Does F2-C benefit from enriched state?
23. Does RAT-C benefit from enriched state?
24. Does learned VQ-style transition code help?
25. Which models form the development Pareto frontier?
26. Which up to three candidates were frozen for held-out?
27. Does 25%->50%->100% train show a clear learning curve?
28. Was an expanded train-only D3 dataset available?
29. Does D3 improve performance?
30. Is the project state-limited, data-limited, both, or neither?
31. Is C18 supported?
32. Is C19 supported?
33. Is C20 supported?
34. Is C21 supported?
35. Is C22 supported?
36. Is READY_FOR_RETARGETING_TEST true?
37. Does the final model remain fast enough for incremental control?
38. Does lift->place improve on identity and continuity simultaneously?
39. What implementation family should Wave27 use?
40. What exact paper claim is now defensible?

---

## 50. Interpretation Rules

If rich causal state strongly improves multiple matched models:

> **The major limitation was incomplete local transition state rather than the language signal itself. Recent latent/action history and phase information are required to interpret the language-conditioned vector field.**

If state-selected prior flow wins:

> **Language-conditioned latent dynamics are easier to model when the flow starts from a causal state-selected transition prior rather than a single global source distribution.**

If temporally coherent flow wins:

> **The endpoint/continuity conflict is reduced by explicitly modeling the latent transition as a temporally coherent path rather than independent future endpoints.**

If retrieval-initialized flow wins:

> **Local transition memory provides a useful source distribution that continuous flow can adapt to the current state and language goal.**

If additional data produces the largest gains:

> **The current architecture was primarily transition-data limited; 257 paired transitions were insufficient to estimate the conditional dynamics reliably.**

If richer state and more data both help:

> **Language-conditioned latent dynamics require both adequate causal phase information and sufficient paired transition coverage.**

If all branches plateau:

> **The next bottleneck is likely the frozen action representation itself, not the choice of transition head.**

In all cases preserve:

> **Changing only the next-goal language causally redirects predicted latent dynamics, including execution-space coordinates.**

---

## 51. Strategic Goal

Wave26 is not another rescue wave.

It is a **factorial implementation study** designed to answer:

```text
What information does the transition model need?
What flow structure should it use?
What temporal objective should it optimize?
How much paired data is required?
```

The intended system remains:

```text
recent latent/action history
+
current action coordinate
+
next language goal
        ↓
causal phase-aware transition distribution
        ↓
local executable latent trajectory
        ↓
continuous robot action
```

Then, once the transition model is strong enough:

```text
new language
-> retarget

STOP / RETURN
-> use stored recoverable trajectory history
```

Wave26 should identify the implementation that makes that next system experiment justified.
