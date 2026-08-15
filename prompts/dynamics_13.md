# Wave 25 Codex Prompt
# Broad Implementation Sweep for Language-Conditioned Latent Dynamics
# From Deterministic Displacement to Multi-Modal, Phase-Aware, and Lightweight Generative Transition Models

## 0. Purpose of Wave 25

Wave 25 must be a **broad implementation-selection wave**.

Do not continue the previous pattern:

```text
one mechanism
-> one intervention
-> one hard gate
-> stop
```

The scientific finding is already stronger than any single implementation:

> **Changing only the next-goal language causally redirects predicted latent dynamics, including execution-space coordinates.**

Wave 25 should therefore explore multiple plausible implementations of the language-conditioned transition operator in parallel and determine which modeling family best converts that already-observed causal language signal into accurate, continuous, executable future trajectories.

The goal is not to rescue one previously failed model.

The goal is to answer:

> **What implementation form is appropriate for language-conditioned latent dynamics?**

---

# 1. Frozen Scientific Facts

Preserve all previous claim decisions.

## Wave 21

Supported components:

```text
Language RedirectGain = 0.250126
95% CI = [0.136495, 0.370798]

Execution RedirectGain = 0.183855
95% CI = [0.100917, 0.263777]
```

Therefore:

> Changing only the next-goal language changes the predicted latent vector field, including its execution dimensions.

C7/C8 remained rejected because endpoint identity, decode/reencode consistency, current-state superiority over prototype, and continuity did not jointly pass.

## Wave 22

Cycle drift is real and correlated with failure.

But global cycle projection:

```text
cycle residual:
2.939692 -> 0.272356
```

damaged target behavior:

```text
RedirectGain -> 0.094419
95% CI crosses zero

endpoint accuracy:
0.516260 -> 0.401423
```

Conclusion:

```text
global decoder support is not sufficient
```

## Wave 23

Goal-specific train geometry is informative:

```text
goal-core margin / endpoint correctness:
Pearson r ≈ 0.715
Spearman rho ≈ 0.844

incremental R^2 beyond global cycle residual ≈ 0.425
```

But static goal-core attraction did not improve endpoint identity.

Conclusion:

```text
static endpoint attraction is too simple
```

## Wave 24

Current-state-conditioned paired transitions contain strong predictive directional information:

```text
full cosine = 0.627467
95% CI [0.599828, 0.655551]

execution cosine = 0.647801
95% CI [0.619503, 0.674799]
```

D2 beats a goal+horizon mean.

However deterministic neighborhood averaging:

```text
underestimates displacement magnitude
degrades endpoint identity
degrades continuity
```

Magnitude recovery is only about 56%–66%.

Conclusion:

```text
the current state contains transition information,
but a single averaged displacement is not a sufficient estimator
```

This motivates Wave 25.

---

# 2. Current Working Scientific Model

Do not force the representation into one of these forms:

```text
language -> static point
language -> static goal core
language -> one global executable manifold
language -> one deterministic displacement
```

The broader working hypothesis is:

```text
p(
    future latent trajectory
    |
    z_previous,
    z_current,
    next language goal,
    horizon
)
```

may be:

```text
multi-modal
heteroscedastic
phase dependent
locally state dependent
temporally structured
```

Wave 25 must compare several implementation families capable of representing these possibilities.

---

# 3. The Final Long-Term System Story

Keep the project-level target in mind.

The intended interface is:

```text
current action latent
+
next atomic language goal
        ↓
language-conditioned latent transition
        ↓
continuous decoded action
```

and later:

```text
new language at any time
-> online retargeting
```

plus:

```text
trajectory history
-> interrupt
-> return to previous recoverable waypoint
```

Wave 25 does not need to demonstrate online retargeting or return yet.

But the selected transition model should be compatible with those future capabilities.

Prefer models that operate incrementally from the current latent rather than regenerate a long trajectory from task start.

---

# 4. High-Level Wave 25 Design

Wave 25 consists of four major phases:

```text
Phase A:
distribution / mode / phase diagnostics

Phase B:
broad development implementation sweep

Phase C:
select 1–2 strongest implementation families

Phase D:
single frozen held-out comparison
```

Do not open held-out during A/B/C.

The held-out test is used only after all implementation choices are frozen.

---

# 5. Frozen Assets

Freeze and hash:

```text
CALVIN action encoder
CALVIN decoder
semantic projection
text encoder
Wave21 B1 LCT
Wave21 B0
normalization statistics

Wave21 session split
Wave21 transition inventory
Wave24 paired-transition dataset
```

No updates to:

```text
representation
encoder
decoder
text encoder
```

during Wave 25.

Write:

`wave25_frozen_manifest.json`

---

# 6. Dataset

Reuse exact paired records:

```text
(z_previous,
 z_current,
 language_goal,
 horizon,
 z_future_h,
 delta_h,
 future action chunk)
```

for:

```text
h in {H1,H2,H4}
```

Keep:

```text
train
development
held-out
```

source-session disjoint.

Expected train scale:

```text
~257 paired transitions
```

Programmatically report exact counts.

Do not increase sample count by treating multiple windows from one physical transition as independent episodes.

---

# 7. Core Causal Inputs

All learned transition models may use only causal information:

```text
z_previous
z_current
delta_previous = z_current - z_previous

next language embedding

horizon embedding
```

Optionally derive train-only/current-only causal features:

```text
||delta_previous||
translation-vs-rotation energy
gripper sign/state from previous/current action chunk
current latent PCA coordinates
```

Do not use:

```text
future latent
future actions
future task labels
future contacts
future simulator state
```

as model input.

---

# 8. Baseline Family 0 — Existing Models

Every development sweep must include:

```text
B0 = Wave21 unconditional predictor
B1 = Wave21 correct-language LCT
B2 = shuffled-language predictor
B3 = null-language intervention

P = language prototype
G = goal+horizon mean displacement
D2 = Wave24 source-conditioned weighted mean
```

These establish the historical reference.

---

# 9. Family 1 — Deterministic Local Regression

Do not assume Wave24 D2 was the best deterministic model.

Try several deterministic local/state-conditioned regressors.

## D1 — 1NN paired displacement

Same `(goal,horizon)`.

Use nearest train source-current latent.

## D2 — existing KNN weighted mean

Frozen historical implementation.

## D3 — local linear regression

For K nearest source states:

fit a ridge local map:

```text
delta_future =
A * [z_current, delta_previous] + b
```

using TRAIN neighbors only.

Regularization candidate set:

```text
alpha ∈ {1e-3, 1e-2, 1e-1, 1}
```

development only.

## D4 — locally weighted affine regression

Weight train neighbors by source-current distance.

## D5 — compact global deterministic MLP

Inputs:

```text
z_previous
z_current
language
horizon
```

Predict:

```text
direction
log magnitude
```

rather than raw displacement.

This tests whether direction/magnitude factorization alone fixes mean shrinkage.

---

# 10. Family 2 — Direction / Magnitude Factorization

Wave24 suggests direction is much easier than magnitude.

Test explicitly.

## F2-A

Predict:

```text
direction = learned
magnitude = goal+horizon mean
```

## F2-B

Predict:

```text
direction = source-conditioned
magnitude = learned
```

## F2-C

Predict both:

```text
direction
log magnitude
```

with separate heads.

## F2-D

Predict execution direction separately from semantic direction.

For example:

```text
u_exec
u_sem
lognorm_exec
lognorm_sem
```

then recombine.

Question:

> Is the failure caused mainly by direction averaging, magnitude averaging, or interference between semantic and execution blocks?

---

# 11. Family 3 — Discrete Multi-Mode Transition Models

Diagnose and model explicit transition modes.

For each `(goal,horizon)` train cell:

cluster normalized displacement directions with:

```text
K ∈ {1,2,3,4}
```

using spherical/cosine clustering.

Then model log magnitude per mode.

Candidate selectors:

## M1 — most frequent mode

Goal+horizon only.

## M2 — nearest-source mode

Mode of nearest train source state.

## M3 — KNN weighted mode vote

K=20.

## M4 — logistic mode selector

Inputs:

```text
z_previous
z_current
delta_previous
language
horizon
```

## M5 — small MLP mode selector

Max:

```text
2 hidden layers
128 units
```

## M6 — mode selector + local residual head

After selecting a mode, predict:

```text
direction residual
log-magnitude residual
```

This should be the strongest compact discrete model.

---

# 12. Family 4 — Compact Mixture Density Network

Train a small conditional mixture model directly.

Call:

```text
MDN
```

Input:

```text
z_previous
z_current
delta_previous
language embedding
horizon embedding
```

Output K components:

```text
pi_k
direction mean/residual
log magnitude mean
log magnitude scale
```

Candidate:

```text
K ∈ {2,3,4}
```

Use development selection.

Primary deterministic evaluation:

```text
argmax component
```

Secondary:

```text
top-2 best-of-N
sampling diversity
```

Do NOT use future target to choose a component except in oracle diagnostics.

Parameter target:

```text
< 500k trainable parameters
```

---

# 13. Family 5 — Small Mixture-of-Experts

Test a compact MoE implementation.

Architecture:

```text
gating network:
(z_previous, z_current, delta_previous, language, horizon)
    ->
expert probabilities

K small deterministic experts:
each predicts direction + log magnitude
```

Candidate experts:

```text
K ∈ {2,3,4}
```

Strong regularization.

Add load-balance diagnostic but do not over-regularize.

Compare:

```text
hard top-1 routing
soft weighted output
```

This is important because soft mixture may recreate mean cancellation, while hard routing may preserve modes.

---

# 14. Family 6 — Lightweight Conditional CVAE

Because the dataset is small, use only a compact latent-variable model.

Call:

```text
cVAE-D
```

Condition:

```text
z_previous
z_current
language
horizon
```

Latent stochastic variable:

```text
r ∈ R^2 or R^4
```

Decoder predicts:

```text
direction + log magnitude
```

Candidate stochastic latent dimensions:

```text
2
4
```

Evaluate:

```text
posterior reconstruction
prior sampling
best-of-N development upper bound
single prior sample
mean of samples
```

Do not let posterior/oracle results count as causal held-out performance.

Purpose:

> Test whether a continuous low-dimensional stochastic latent captures transition modes better than discrete mixtures.

---

# 15. Family 7 — Lightweight Conditional Flow Matching

A small flow-style model is allowed in Wave25 as an implementation exploration.

Do NOT train a large policy.

Model only:

```text
latent displacement delta
```

not robot images/actions end to end.

Call:

```text
Latent-CFM
```

Condition:

```text
z_previous
z_current
delta_previous
language
horizon
```

Target distribution:

```text
delta_future
```

Use a small MLP vector field.

Keep parameter count comparable to MDN/MoE where possible.

Train on TRAIN only.

Development inference:

```text
8 ODE steps
```

and optionally:

```text
16 ODE steps
```

as a compute sensitivity diagnostic.

Do not sweep architecture extensively.

Question:

> Does continuous distributional transport model the displacement family better than discrete modes?

---

# 16. Family 8 — Lightweight Conditional Diffusion in Displacement Space

A small diffusion model is also allowed as a development-only implementation branch.

Call:

```text
Latent-Diff
```

Model only 32-D displacement or 16-D execution displacement + deterministic semantic component.

Condition on:

```text
z_previous
z_current
delta_previous
language
horizon
```

Use a compact MLP denoiser.

Small diffusion schedule only.

For example:

```text
training noise steps = 20–50
inference DDIM steps = 8–16
```

Do not build a large transformer/diffusion policy.

Primary reason for including this family:

> If discrete mixture models fail but a small continuous generative model succeeds, the transition structure may be continuous/multi-branched rather than cleanly clustered.

---

# 17. Family 9 — Retrieval-Augmented Transition Model

Because data are small, test a hybrid nonparametric + learned method.

Call:

```text
RAT = Retrieval-Augmented Transition
```

Steps:

1. retrieve K=20 same-goal/horizon source transitions;
2. feed query state + retrieved displacement statistics into a small network;
3. predict mixture weights or a residual over retrieved candidates.

Variants:

```text
RAT-A:
attention over retrieved displacement vectors

RAT-B:
top-k candidate scoring

RAT-C:
retrieve then residual-correct selected displacement
```

This may be especially appropriate at current dataset scale.

---

# 18. Family 10 — Phase-Augmented Models

Wave25 should test whether causal phase information is the missing variable.

Available causal proxy:

```text
delta_previous = z_current - z_previous
```

Additional causal features from current/history action chunks:

```text
gripper state/sign
translation norm
rotation norm
action speed
current-vs-previous latent velocity angle
distance from known next annotation onset = 0 at transition boundary
```

Do NOT use future contact or future state.

For the strongest models from Families 3–9 compare:

```text
without phase proxy
vs
with phase proxy
```

If phase features materially improve performance, this becomes an important scientific result.

---

# 19. Train-Only Distribution Diagnostics

Before comparing models, generate a detailed train-only report.

For each goal/horizon:

```text
direction cluster count
direction effective rank
log-magnitude distribution
mode entropy
source-session support
pairwise cosine histogram
norm histogram
```

Also compute:

```text
cancellation ratio =
||mean displacement|| / mean ||displacement||
```

and local cancellation ratio for Wave24 neighborhoods.

Test whether cancellation predicts:

```text
magnitude underestimation
endpoint failure
continuity error
```

---

# 20. Oracle Suite

Use several oracle diagnostics to separate representation capacity from selector capacity.

## O1 — oracle train displacement

Choose the closest true TRAIN displacement to the development target.

## O2 — oracle mode

Choose the best train-derived mode.

## O3 — oracle retrieved neighbor

Choose the best among K retrieved source-conditioned transitions.

## O4 — best-of-N generative sample

For CVAE/flow/diffusion, choose best among N samples using development ground truth.

Use:

```text
N = 8
```

These are upper bounds only.

They must never be counted as causal model performance.

Key interpretation:

```text
strong oracle + weak non-oracle
=> selection/state-information problem

weak oracle
=> representation/model-family support problem
```

---

# 21. Development Evaluation Metrics

Every family should report, where applicable:

```text
H1/H2/H4 full latent MSE
H1/H2/H4 execution MSE
H1/H2/H4 semantic MSE

displacement cosine
execution cosine

norm ratio
absolute norm error

H4 decoded action MSE

endpoint macro accuracy
decode/reencode macro accuracy

continuity error

cycle residual

goal target margin

current-state dependence

language RedirectGain
execution RedirectGain
```

Distributional models additionally report:

```text
NLL / ELBO where meaningful
mode entropy
sample diversity
best-of-N gap
argmax-vs-sampled gap
```

---

# 22. Development Pareto Analysis

Do not reduce model selection to one metric.

Construct a development Pareto table over:

```text
H2 full MSE
H4 decoded MSE
endpoint identity
decode/reencode identity
continuity
RedirectGain
Execution RedirectGain
parameter count
```

Identify models that are Pareto dominated.

Discard only clearly dominated implementations.

Keep 1–2 final candidates if they represent different model families and both are competitive.

---

# 23. Development Minimum Requirements

A candidate is eligible for held-out only if it satisfies:

```text
RedirectGain > 0
Execution RedirectGain > 0

H2 full MSE < Wave24 D2

H4 decoded MSE < Wave24 D2

endpoint accuracy > Wave24 D2

continuity < Wave24 D2
```

This is deliberately broader than earlier strict gates.

Do NOT require endpoint >=0.60 at development.

Do NOT require decode/reencode >=0.60 yet.

The purpose is implementation selection.

---

# 24. Preserve Wave21 Language Effect

A candidate should preferably retain:

```text
>= 75% of Wave21 full RedirectGain
>= 75% of Wave21 execution RedirectGain
```

Models below 75% remain descriptive but are not preferred for final selection.

Use 75% because Wave25 explores new probabilistic models and should not over-constrain implementation discovery.

---

# 25. Final Candidate Selection

Select at most:

```text
2 models
```

for held-out.

Recommended:

- best compact/discrete model;
- best continuous/generative model;

IF both satisfy development minimums.

If only one family qualifies, select one.

Freeze selection rule before seeing held-out.

Suggested lexicographic selection:

```text
1. H4 decoded MSE
2. H2 full MSE
3. endpoint macro accuracy
4. continuity
5. decode/reencode accuracy
6. parameter efficiency
```

Write:

`wave25_final_candidate_selection.json`

---

# 26. Held-Out Preregistration

Before materializing held-out arrays, freeze:

```text
selected 1–2 models
all architecture configs
all seeds
all checkpoints
sampling procedure
number of samples
ODE/DDIM steps
mode count
retrieval K
phase features
metrics
bootstrap seed
claims
```

Write:

`wave25_final_test_preregistration.json`

Only then open held-out.

---

# 27. Held-Out Claim: C15

Define:

```text
C15_distributional_language_conditioned_transition
```

C15 is SUPPORTED if at least one selected model satisfies on held-out:

## G1
Full RedirectGain lower95 > 0.

## G2
Execution RedirectGain lower95 > 0.

## G3
H2 full MSE improves over Wave24 D2 with favorable CI.

## G4
H4 decoded MSE improves over Wave24 D2 with favorable CI.

## G5
Endpoint macro accuracy improves over Wave24 D2.

## G6
Continuity improves over Wave24 D2.

## G7
Current-state dependence remains: same language/horizon produces different transition predictions for meaningfully different current states, and the model beats goal+horizon prior/mean.

This claim does NOT require 0.60 endpoint accuracy.

It establishes that distributional/state-aware modeling is superior to deterministic averaging.

---

# 28. Stronger Held-Out Claim: C16

Define:

```text
C16_executable_language_conditioned_transition_modes
```

C16 requires C15 plus:

```text
endpoint macro >= 0.60
decode/reencode macro >= 0.60
continuity <= Wave21 B1
positive target/mode margin on >=5/6 goals
```

and canonical:

```text
lift_blue_block_slider -> place_in_slider
```

must improve on both:

```text
H2 latent error
H4 decoded action error
```

C16 is stronger but not mandatory.

---

# 29. Stronger Mechanistic Claim: C17

If a probabilistic model passes and the following diagnostics hold:

```text
mean cancellation predicts D2 failure

oracle mode/sample clearly beats deterministic mean

non-oracle conditional model closes significant oracle gap

language changes conditional mode/sample distribution

current state changes conditional mode/sample distribution
```

then support:

```text
C17_language_and_state_shape_transition_distribution
```

Safe wording:

> **Language and current state jointly shape a conditional distribution over future latent transitions; deterministic averaging loses this structure.**

---

# 30. Same-State Language Swap

For every selected model repeat Wave21's strongest causal intervention.

Same:

```text
z_previous
z_current
horizon
weights
random seed schedule
```

Change only language.

For probabilistic models, record:

```text
distribution parameters
mode probabilities
sample set
mean/sample trajectories
selected deterministic trajectory
```

Measure whether language changes:

```text
mode probabilities
flow endpoint distribution
diffusion sample distribution
predicted displacement
```

This should become one of the central paper figures.

---

# 31. Same-Language Different-State Test

Fix:

```text
language
horizon
```

vary current states.

Measure:

```text
distribution shift
mode probability shift
direction shift
magnitude shift
sample diversity shift
```

This directly tests:

> language selects the transition family; current state shapes the local member/distribution.

---

# 32. Retargeting Compatibility Diagnostic

Do not run full closed-loop retargeting yet, but test model compatibility offline.

For a sequence:

```text
z_current
goal A
predict one latent step -> z_A1
switch goal to B
predict next step from z_A1
```

Compare against a model that keeps A.

Question:

> Can the transition model be queried incrementally after a language switch without requiring trajectory regeneration from the initial task state?

Use development only.

This is a bridge toward future online retargeting.

---

# 33. History / Return Compatibility Diagnostic

For selected models, store generated waypoints:

```text
z_0, z_1, ..., z_k
```

No physical return execution yet.

Test only:

```text
decoder reconstruction
waypoint recoverability
distance between stored waypoint and E(D(z))
```

Purpose:

Ensure the chosen transition representation can later support `return-to-history`.

Do not claim reversibility.

---

# 34. Canonical Lift-to-Place Experiment

For every eligible development/held-out example:

```text
lift_blue_block_slider -> place_in_slider
```

compare:

```text
Wave21 B1
Wave24 D2
best deterministic family
best discrete/mode family
best generative family
oracle
ground truth
```

At H1/H2/H4 report:

```text
direction cosine
execution cosine
norm ratio
full MSE
execution MSE
decoded MSE
endpoint identity
decode/reencode identity
continuity
distribution/mode diagnostics
```

No cherry-picking.

---

# 35. Required Outputs by Model Family

For every implementation save a common JSON schema:

```text
model_family
model_name
parameter_count

train_metrics
dev_metrics

H1
H2
H4

full_mse
execution_mse
semantic_mse
decoded_mse
endpoint_accuracy
decode_reencode_accuracy
continuity
redirect_gain
execution_redirect_gain

runtime
memory

selection_status
```

This makes cross-family comparison easy.

---

# 36. Computational Budget

Because Wave25 is broad, keep models small.

Guideline:

```text
deterministic / selector models:
< 500k params

MDN / MoE:
< 1M params

CVAE / flow / diffusion:
prefer < 1–2M params
```

Use early stopping on TRAIN/DEVELOPMENT only if preregistered.

Do not run huge architecture sweeps.

The goal is to compare modeling principles.

---

# 37. Statistical Protocol

Independent unit:

```text
continuous source session
```

Use:

```text
10,000 bootstrap replicates
seed = 250825
```

Use paired comparisons where predictions correspond to the same transition.

Same-state language swaps remain paired within boundary.

---

# 38. Failure Interpretation Matrix

If:

```text
oracle discrete mode strong
non-oracle discrete weak
```

interpret:

```text
mode structure exists
selector lacks causal state information
```

Next direction:

```text
phase/contact-conditioned state
```

If:

```text
discrete oracle weak
continuous generative oracle/sample strong
```

interpret:

```text
transition distribution is continuous/multi-branched,
not well captured by discrete modes
```

Next direction:

```text
latent diffusion / flow
```

If:

```text
all probabilistic methods weak
but deterministic B1 remains strongest
```

interpret:

```text
data scale may be too small
or frozen representation may not expose transition phase
```

Next direction:

```text
more transition data
or representation enrichment
```

If:

```text
phase features strongly improve all model families
```

interpret:

```text
the key missing variable is phase/history information
```

Next direction:

```text
phase-aware latent state
```

Do not respond by shrinking the Wave21 causal language claim.

---

# 39. Required Files

Produce:

```text
twenty_fifth_wave_results.md
twenty_fifth_wave_next_experiment.md

wave25_frozen_manifest.json
wave25_dataset_audit.md

wave25_distribution_diagnostics.md
wave25_direction_modes.json
wave25_magnitude_modes.json
wave25_cancellation_analysis.md
wave25_oracle_suite.md

wave25_deterministic_family_results.md
wave25_factorized_direction_magnitude_results.md
wave25_discrete_mode_results.md
wave25_mdn_results.md
wave25_moe_results.md
wave25_cvae_results.md
wave25_flow_results.md
wave25_diffusion_results.md
wave25_retrieval_augmented_results.md
wave25_phase_augmented_results.md

wave25_development_pareto.csv
wave25_final_candidate_selection.json

wave25_model_preregistration.json
wave25_seed_preregistration.json
wave25_final_test_preregistration.json

wave25_heldout_results.md
wave25_same_state_language_swap.md
wave25_same_language_different_state.md
wave25_retargeting_compatibility.md
wave25_history_return_compatibility.md
wave25_lift_to_place_case.md

wave25_claim_decision.json
wave25_failure_taxonomy.md
wave25_statistical_report.md

wave25_future_implementation_plan.md

publication_tables/
publication_figures_data/

exact_commands.sh
environment_freeze.txt
files_changed.txt
tests_report.txt

updated_RESEARCH_LOG.md
updated_NEXT_EXPERIMENT.md
```

If a family is not run due to infrastructure/dependency failure, create its report and state why.

---

# 40. Required Tests

At minimum:

```text
all frozen representation hashes unchanged
session split unchanged
held-out masked until final freeze

all model inputs causal
future arrays excluded from selectors

direction normalization correct
log magnitude finite

train-only clustering
train/dev only model selection

oracle future information isolated
oracle never used as causal selector

same-state language intervention changes only language

same-language different-state test preserves language

distribution samples reproducible from frozen seeds

all JSON valid
all outputs finite

bootstrap by source session
10,000 replicates
seed 250825
```

---

# 41. What Not to Do

Do NOT:

```text
reopen DEL
reopen old F2 rescue
add static goal-core attraction
add global cycle projection as primary fix
claim multimodality before diagnostics
claim attractors before stability evidence
run full closed-loop retargeting before transition model selection
claim strict reversibility
claim VLA/WAM cannot replan
```

---

# 42. Final Claim Decision JSON

Write:

`wave25_claim_decision.json`

with fields:

```text
C15_distributional_language_conditioned_transition
C16_executable_language_conditioned_transition_modes
C17_language_and_state_shape_transition_distribution

deterministic_local_regression_best
direction_magnitude_factorization_best
discrete_modes_supported
MDN_supported
MoE_supported
CVAE_supported
flow_supported
diffusion_supported
retrieval_augmented_supported
phase_features_help

oracle_discrete_gap
oracle_generative_gap
nonoracle_closes_oracle_gap

language_redirect_preserved
execution_redirect_preserved
current_state_matters
endpoint_identity_improved
decode_reencode_improved
continuity_improved

recommended_wave26_family
```

---

# 43. Final Report Questions

The final report must answer:

1. Exact train/dev/test transition counts?
2. How many stable direction modes exist?
3. How many magnitude regimes exist?
4. Does mode entropy predict deterministic cancellation?
5. Which deterministic local model is best?
6. Does direction/magnitude factorization help?
7. Does nearest-mode selection help?
8. Does KNN mode voting help?
9. Does logistic selection help?
10. Does small MLP selection help?
11. Does mode + residual help?
12. Does MDN help?
13. Does hard MoE beat soft MoE?
14. Does cVAE help?
15. Does lightweight flow help?
16. Does lightweight diffusion help?
17. Does retrieval augmentation help?
18. Do causal phase features help?
19. Which family has best H2 full MSE?
20. Which has best H4 decoded MSE?
21. Which has best endpoint identity?
22. Which has best decode/reencode identity?
23. Which has best continuity?
24. Which preserves the most RedirectGain?
25. Does an oracle discrete mode provide a strong upper bound?
26. Does best-of-N cVAE/flow/diffusion provide a stronger upper bound?
27. How much oracle gap does the best causal model close?
28. Does current state change the selected distribution under fixed language?
29. Does changing language change the selected distribution under fixed state?
30. Does the selected model remain compatible with incremental retargeting?
31. Are generated waypoints suitable for future return-to-history tests?
32. Does lift -> place improve?
33. Is C15 supported?
34. Is C16 supported?
35. Is C17 supported?
36. Which modeling family should Wave26 pursue?
37. What exact paper claim is now defensible?

---

# 44. Interpretation

The key philosophy of Wave25 is:

> **Do not keep weakening the scientific question because one implementation fails. Search broadly for the transition parameterization that best realizes the already-observed language-conditioned vector field.**

The implementation ladder is:

```text
deterministic local regression
        ↓
direction/magnitude factorization
        ↓
discrete transition modes
        ↓
mixture-density / MoE
        ↓
retrieval-augmented prediction
        ↓
compact CVAE
        ↓
lightweight conditional flow
        ↓
lightweight latent diffusion
        ↓
phase-aware transition state
```

Wave25 should compare these routes systematically.

The intended final conceptual model remains:

```text
current latent state
+
next language goal
+
recent latent history
        ↓
conditional distribution over local future transitions
        ↓
select / sample an executable transition
        ↓
decode continuous action
```

Future system layer:

```text
new language at any time
-> retarget

stored waypoint history
-> interrupt / return
```

Wave25's job is to determine the best implementation basis for that system.
