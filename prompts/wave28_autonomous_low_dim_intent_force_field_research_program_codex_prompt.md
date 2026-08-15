# Wave 28+ Autonomous Research Program Prompt
# Low-Dimensional Intent Force-Field Adapter for Editable Action Coordinates
# Broad architecture/loss/composition search + automatic Wave29/Wave30 iteration with full audit trail

## 0. Role and mission

You are Codex operating inside the existing **Actions as Coordinates** research repository.

Your task is not merely to run one Wave28 experiment. Your task is to execute a **disciplined multi-wave research program** beginning at Wave28, with the goal of identifying a defensible implementation of:

> **A frozen action-language latent policy augmented by a small, low-dimensional intent force-field adapter that supports online intention retargeting while preserving the underlying action representation and behavior.**

You must search broadly across:

```text
intent encoding
force-field state dimension
field dynamics
low-rank composition
state conditioning
loss functions
behavior-preservation constraints
continuity constraints
redirect objectives
return/cycle objectives
subspace initialization
adapter rank
linear/nonlinear composition
F1/F2 behavioral backbones
```

If Wave28 does not produce a convincing result, you are explicitly authorized to design and execute Wave29, Wave30, and subsequent waves **without asking for confirmation**, provided that:

1. every new wave is scientifically motivated by the previous wave;
2. no held-out result is used to retroactively modify the same wave;
3. every wave gets its own frozen prompt, preregistration, report, claim decision, and next-experiment document;
4. previous failures are preserved rather than overwritten;
5. no future information is leaked into inference;
6. no scientific claim is upgraded without evidence.

The goal is not to make one arbitrary model pass. The goal is to determine **what encoding form, field dynamics, composition operator, loss combination, and ablation structure are actually required** for the full Actions-as-Coordinates retargeting story.

---

## 1. Read project context first

Before changing code, read the project-level story file if present:

```text
PROJECT_STORY.md
PAPER_STORY.md
actions_as_coordinates_project_paper_outline.md
```

Also read all Wave21–Wave27 prompts/reports/claim decisions that exist. At minimum reconstruct the current scientific state:

```text
Wave21:
changing only next-goal language causally redirects latent dynamics
including execution-space coordinates.

Wave22:
global decoder-cycle projection reduces cycle residual
but damages target identity.

Wave23:
goal-specific geometry predicts identity,
but static goal-core attraction does not repair it.

Wave24:
source-conditioned paired transitions predict displacement direction,
but deterministic averaging shrinks magnitude and hurts identity/continuity.

Wave25:
66-model broad implementation sweep;
continuous flow promising;
no development candidate passed all criteria.

Wave26:
79 development entries;
RAT-C / Prior-CFM / History-CFM held-out;
higher-dimensional history alone did not help;
data scaling signal appeared.

Wave27:
new independent transition data collected;
language redirect becomes very strong on prospective data;
retrieval remains strong;
trajectory accuracy/continuity still insufficient for retargeting readiness.
```

Do not rewrite these historical decisions.

---

## 2. Frozen core scientific claim

The central result to preserve is:

> **Changing only the next-goal language causally redirects predicted latent dynamics, including execution-space coordinates.**

The new research question is no longer `Can language affect the latent?` It is:

> **Can a tiny, low-dimensional, learnable latent control field convert a frozen language-addressable action representation into an editable intention state that can be redirected online without relearning the action model?**

---

## 3. Main method hypothesis

Freeze the original action-text representation. Do not modify:

```text
action encoder
text encoder
action VAE / latent representation
action decoder
semantic projection
```

unless a later wave explicitly concludes that the frozen representation is the bottleneck.

The new mechanism is a low-dimensional intention state `q_t ∈ R^k`, with initial sweep `k ∈ {1,2,4,8}`. The conceptual architecture is:

```text
z_base_t = frozen behavioral latent / frozen backbone prediction
q_t      = low-dimensional intention-field state
delta_z  = adapter(q_t, z_base_t, causal state)
z'_t     = z_base_t + delta_z
a_t      = D_frozen(z'_t)
```

The adapter is the only new intervention pathway. The frozen decoder must receive the modified original latent dimension. Do **not** append new dimensions that the frozen decoder never learned to consume.

---

## 4. Audit the true backbone interface

Do not assume the frozen action encoder accepts robot observation/state. The existing representation is action-chunk based. Audit the exact repository implementation and write `wave28_backbone_interface_audit.md` containing:

```text
what produces z_base
which tensors are frozen
which tensors contain language
which tensors contain action
which tensors contain state/history
decoder input dimensionality
F1 inputs/outputs
F2 inputs/outputs
Wave21 B1 interface
Wave27 RAT-C interface
```

Possible valid bases include:

```text
current executed latent z_t
F1 prediction
F2-refined prediction
Wave21 language transition prediction
Wave27 RAT-C/retrieval prediction
```

Use what the repository actually supports.

---

## 5. Separation of responsibilities

The target method should preserve this conceptual separation:

```text
F1 / F2 / frozen behavioral backbone:
what local behavior is plausible and how action evolves

low-dimensional intent force-field:
how intention changes when language changes

frozen decoder:
how modified action coordinates become robot actions
```

Do not let the force-field adapter quietly relearn the entire policy. Track adapter parameter count and percentage of total parameters trained.

---

## 6. Two distinct spaces

Do not conflate the original action latent `z` with the new low-dimensional intention/control space `q`.

Language may map to a low-dimensional coordinate `p(h)=Wh`. Any attractor-like interpretation should initially apply only to `q`-space. Do **not** assume that each language corresponds to a static attractor in the original 32-D action latent; Waves22–23 already showed that simple static endpoint attraction is inadequate.

Preferred terminology:

```text
intention coordinate
latent control field
language-conditioned intention field
low-rank residual steering
```

---

## 7. Force-field formulations to compare

Wave28 must compare multiple forms, not one preselected formula.

### FF0 — no adapter
Frozen backbone only.

### FF1 — direct low-rank language difference

```text
d = p(h_target) - p(h_current)
delta_z = B d
```

### FF2 — accumulating linear field

```text
q_{t+1} = q_t + alpha * (p_target - p_current)
delta_z_t = B q_t
```

### FF3 — target-attractor field

```text
q_{t+1} = q_t + alpha * G * (p_target - q_t)
```

### FF4 — state-conditioned attractor field

```text
q_{t+1} = q_t + alpha_t * G_theta(z_t, causal_state) * (p_target - q_t)
```

### FF5 — nonlinear residual field

```text
q_{t+1} = q_t + f_theta(q_t, p_target, z_t, causal_state)
```

Keep `f_theta` small.

### FF6 — potential-style q-field
Learn a compact scalar `U_theta(q,p_target,state)` and update along `-∇_q U`. Never call this physical energy.

### FF7 — gated field

```text
g_t in [0,1]
delta_z = g_t * B q_t
```

### FF8 — velocity-conditioned field
Use causal recent latent velocity / previous displacement.

### FF9 — retrieval-conditioned field
Retrieve similar historical intention transitions and use them to initialize or bias q-field direction.

---

## 8. q-dimension sweep

For major field families evaluate:

```text
k = 1, 2, 4, 8
```

Do not assume 2-D is correct. Measure:

```text
retarget quality
base-policy preservation
continuity
return symmetry
parameter count
effective dimension actually used
```

If q collapses to lower rank, report it.

---

## 9. Language-to-intention encodings

Compare:

### E0 — frozen text embedding + linear projection
`p(h)=Wh`

### E1 — normalized linear projection
`p(h)=normalize(Wh)` plus learned scale.

### E2 — small MLP projection
At most two layers.

### E3 — pairwise relative encoder

```text
d(h_current,h_target) = MLP([h_current,h_target,h_target-h_current])
```

### E4 — antisymmetric pair encoder
Encourage `d(h0,h1) ≈ -d(h1,h0)`.

### E5 — train-derived action/text initialization
Initialize W from existing action-text alignment geometry using TRAIN only.

### E6 — learned intention dictionary
Each atomic goal has a small q-anchor initialized from text, then trainable. Compare against fully text-derived coordinates.

---

## 10. Subspace / adapter forms

Compare how q affects z.

### C0 — fixed random low-rank B
Random orthogonal basis; train q encoder only.

### C1 — PCA/SVD initialized B
Initialize from TRAIN transition residual/displacement directions.

### C2 — learned global B
Train `B ∈ R^(d_z × k)` with norm/orthogonality controls.

### C3 — block-separable B
Separate semantic/execution projection.

### C4 — execution-only B

### C5 — semantic-only B
Negative control.

### C6 — state-dependent low-rank B(z)
Predict a small rotation/scaling of B while rank remains <=k.

### C7 — hypernetwork low-rank adapter
Tiny hypernetwork predicts low-rank factors.

### C8 — nonlinear residual adapter
`delta_z=A_theta(q,z)` under matched parameter budget.

### C9 — full-rank residual MLP control
Critical matched control. If full-rank wins clearly, report evidence against strict low-rank sufficiency.

---

## 11. Composition operators

Compare:

```text
COMP0 additive: z' = z + delta_z
COMP1 gated additive: z' = z + g*delta_z
COMP2 normalized residual: scale relative to train latent/residual stats
COMP3 FiLM-style modulation on selected latent dimensions
COMP4 low-rank rotation-like transform
COMP5 convex interpolation with an adapted dynamic target, control only
```

Do not revive Wave23 static goal-core attraction.

---

## 12. Behavioral backbone conditions

Compare, when interface-compatible:

```text
B0 frozen F1 only
B1 frozen F2 only
B2 F1 + F2 refinement
B3 Wave21 B1 transition backbone
B4 Wave27 RAT-C base
B5 best prospective retrieval transition base
```

Question: is the intent-field adapter a reusable retargeting layer, or only a patch for one base?

---

## 13. Ordered retarget-event dataset

Construct explicit events:

```text
(z_previous,
 z_current,
 h_current,
 h_target,
 time_since_instruction_change,
 H1/H2/H4 future latents/actions,
 causal physical state if available)
```

Use Wave21 ordered annotation transitions and Wave27 new prospective transitions. Maintain source-session separation.

Do not add `is_new_instruction` if sequence order already determines the event. `time_since_instruction_change` is allowed if available online.

---

## 14. No-switch anchor data

Create matched events with `h_target=h_current`. Correct adapter behavior should be neutral or near-neutral:

```text
small q change
small delta_z
frozen-base behavior preserved
```

This is mandatory to prove the retargeting layer does not destroy normal behavior.

---

## 15. Return-pair construction

Build language/intention pairs:

```text
h0 -> h1
h1 -> h0
```

Wave28 initially tests **intention-space return**, not strict physical reversibility.

Primary targets:

```text
q moves back toward the h0 intention coordinate
adapter-modified z becomes compatible with h0 local behavior
```

Do not require an `is_return` token.

---

## 16. Loss library

Implement a reusable library.

### L0 — behavioral latent prediction / teacher consistency

### L1 — true frozen-decoder action loss

```text
L_decode = MSE(D_frozen(z_pred), action_gt)
```

Decoder parameters frozen, gradient must flow through decoder into adapter.

### L2 — redirect direction loss
Align adapter-induced change with observed target-transition residual/direction.

### L3 — q-target field loss
Encourage q to move toward `p(h_target)` in q-space only.

### L4 — continuity loss
Penalize latent/action jumps beyond TRAIN ground-truth statistics.

### L5 — no-switch anchor loss
Preserve frozen backbone when target=current.

### L6 — behavior-preservation distillation

### L7 — antisymmetry loss
`d(h0,h1) ≈ -d(h1,h0)`.

### L8 — intention return-cycle loss
After `h0->h1->h0`, q returns near its initial intention coordinate. This is not decoder cycle consistency.

### L9 — perturbation norm regularization

### L10 — basis orthogonality/diversity

### L11 — execution-block preference, optional soft regularizer

### L12 — transition contrastive loss
Correct target transition vs wrong-language matched-state transitions.

### L13 — field smoothness/Jacobian control

### L14 — sparse/intervention gate regularizer
Use only if evidence supports gating.

---

## 17. Structured loss combinations

Do not brute-force arbitrary subsets.

```text
Group A: L0 + L2 + L5
Group B: A + L4
Group C: A + L1
Group D: A + L4 + L1
Group E: D + L7 + L8
Group F: D + L3
Group G: D + L12
Group H: best above + L9 + optional L10
```

Development may add at most two auxiliary losses beyond the selected base group.

Normalize losses using TRAIN-scale statistics first. For uncertain multipliers use only a small set such as `{0.1,0.3,1.0}`. Never modify weights after held-out opens.

---

## 18. Return symmetry at four levels

Keep these distinct:

```text
R-INTENT: q returns
R-LATENT: adapted z returns toward prior intention-conditioned region/path
R-ACTION: decoded behavior returns toward prior action behavior
R-PHYSICAL: robot returns to a previously visited recoverable physical state
```

Wave28 primarily targets R-INTENT and R-LATENT. R-ACTION may be offline. R-PHYSICAL belongs to later matched-state closed-loop experiments.

---

## 19. Metrics

Primary retarget metrics:

```text
RedirectGain
Execution RedirectGain
H1/H2/H4 latent error
H4 decoded action MSE
endpoint identity
decode/reencode identity
continuity
action jump at instruction switch
response latency after instruction change
no-switch base-policy degradation
adapter norm
```

Field metrics:

```text
q effective rank
q path length
distance to target intention coordinate
field speed/curvature
Jacobian norm
local convergence around p(h)
```

Return metrics:

```text
q return error
latent return error
decoded-action return error
forward/reverse direction cosine
antisymmetry error
return continuity
```

Do not call the field an attractor unless multiple initial states show repeated stable convergence behavior.

---

## 20. Wave28 staged tournament

Do not run the full Cartesian product.

### Stage 1 — interface/sanity

Verify:

1. frozen backbone reproduction;
2. decoder gradient-through with frozen decoder params;
3. q=0/no-adapter reproduces base;
4. random low-rank perturbation measurably changes decoded action;
5. B numerical rank;
6. language swap changes p(h);
7. antisymmetric encoder behavior when enabled;
8. no future data enters q update.

### Stage 2 — q dimension × subspace × simple field

Compare k `{1,2,4,8}` with:

```text
random B
PCA B
learned B
block-separable B
FF1 / FF3 / FF4
additive / gated additive
```

using minimal loss Groups A/D.

### Stage 3 — field dynamics

For non-dominated candidates compare:

```text
FF3 attractor
FF4 state-conditioned attractor
FF5 nonlinear
FF7 gated
FF8 velocity-conditioned
FF9 retrieval-conditioned
```

### Stage 4 — composition

Compare:

```text
additive
gated additive
normalized residual
FiLM
state-dependent low-rank
full-rank matched control
```

### Stage 5 — loss tournament

Compare Groups A–H.

### Stage 6 — backbone generality

Apply strongest adapters across F1, F1+F2, and best available language/retrieval backbone where compatible.

### Stage 7 — ordered retarget/return evaluation

Run sequences such as:

```text
lift -> place
place -> lift
turn_on -> turn_off
turn_off -> turn_on
h0 -> h1 -> h0
```

without a dedicated return flag.

---

## 21. Minimum required ablation table

Include at least:

```text
Frozen base, no adapter

1D learned B
2D learned B
4D learned B
8D learned B

2D random B
2D PCA B
2D learned B

2D semantic-only B
2D execution-only B
2D block-separable B

2D static residual
2D attractor field
2D state-conditioned field
2D nonlinear field
2D gated field
2D retrieval-conditioned field

2D additive
2D gated-additive
2D normalized
2D FiLM

2D no direction loss
2D no continuity
2D no anchor
2D no decoder loss
2D no return-cycle
2D full loss

2D F1
2D F2
2D F1+F2

2D full-rank matched-capacity control
```

Add scientifically motivated rows if needed.

---

## 22. Development scorecard and Pareto analysis

No one all-or-nothing gate. For every candidate report:

```text
REDIRECT
EXEC_REDIRECT
PREDICTION
DECODE
IDENTITY
RECODE
CONTINUITY
BASE_PRESERVATION
RETURN_INTENT
RETURN_LATENT
PARAM_EFFICIENCY
```

Classify relative to frozen base / Wave21 B1 / Wave27 RAT-C where applicable:

```text
strongly improved
improved
neutral
worse
strongly worse
```

Construct Pareto fronts across:

```text
RedirectGain
Execution RedirectGain
H2 full MSE
H4 decoded MSE
endpoint identity
continuity
base preservation
return error
parameter count
```

Select up to four final Wave28 candidates from distinct mechanisms. Freeze before held-out.

---

## 23. Wave28 independent claims

Do not force one master pass/fail.

```text
C30 low-dimensional adapter preserves frozen behavior
C31 low-dimensional intention field improves language retargeting
C32 dynamic q-field outperforms static residual steering
C33 learned low-rank subspace outperforms random/PCA controls
C34 q-space return symmetry is supported
C35 continuity/anchor losses improve editability without destroying redirect
C36 adapter generalizes across frozen behavioral backbones
```

Each must be:

```text
SUPPORTED
NOT_SUPPORTED
MIXED
NOT_TESTED
```

---

## 24. Story-success criteria

The core method becomes substantially defensible if held-out/prospective evidence supports:

### S1 — frozen behavior remains intact
No-switch examples stay close to the original policy.

### S2 — language change induces adapter movement
Changing target language at fixed state changes q and adapted z in a target-specific manner.

### S3 — low-dimensional control is sufficient
A small-k adapter is competitive with a full-rank residual control.

### S4 — dynamic field matters
Dynamic q evolution outperforms one-shot static residual/interpolation.

### S5 — continuity is maintained
Retargeting does not create unacceptable decoded action jumps.

### S6 — return symmetry exists in intention space
`h0->h1->h0` reverses q-intention transfer without a dedicated return token.

### S7 — behavioral backbone remains primary
Adapter is parameter-efficient and does not relearn the action model.

These criteria matter more than forcing one exact architecture.

---

## 25. Engineering readiness

Define a separate flag:

```text
READY_FOR_CLOSED_LOOP_RETARGET
```

Before held-out, derive numerical thresholds from TRAIN/DEV for:

```text
positive full/execution RedirectGain confidence
endpoint/recode improvement
continuity tolerance
no-switch degradation tolerance
return-intent error
latency
```

Do not invent thresholds after held-out.

---

## 26. Automatic Wave29 if Wave28 is insufficient

If Wave28 does not satisfy the story-success criteria, do not stop at "failed". Generate and save `prompts/dynamics_wave29.md`, then execute Wave29.

Wave29 must target the actual diagnosed failure. Examples:

### Low-rank capacity insufficient
Try:

```text
k=8/16
piecewise low-rank fields
mixture of low-rank adapters
state-selected basis
```

### q-space works but projection B fails
Try:

```text
state-dependent B
hypernetwork low-rank
decoder-Jacobian-aware initialization
semantic/execution block-specific basis
```

### Redirect works but continuity fails
Try:

```text
continuous-time q ODE
smaller field integration step
q velocity/damping state
trajectory-level decoder loss
data-calibrated damping
```

### Continuity works but identity fails
Try:

```text
transition contrast
antisymmetric pair encoding
retrieval-conditioned q targets
state-conditioned field geometry
```

### Return symmetry fails
Try:

```text
explicit antisymmetric pair encoder
invertible q dynamics
shared forward/reverse field
reversible neural ODE in q-space
```

### No-switch behavior degrades
Try:

```text
zero-initialized adapter
hard anchor gate
LoRA-style zero scaling
phase-dependent intervention gate
```

### F1/F2 base mismatch dominates
Test different frozen behavioral bases rather than changing the action representation immediately.

---

## 27. Automatic Wave30+ policy

If Wave29 remains insufficient, generate Wave30. Do not merely increase epochs or network size. Every wave must target a diagnosed mechanism.

Legitimate future directions include:

```text
invertible intention dynamics
neural controlled differential equation in q
mixture-of-fields
state-selected attractor family
retrieval-seeded q trajectories
decoder-Jacobian-aware low-rank control
contact-conditioned field
continuous-time field integration
online learned gain alpha_t
```

Every next wave must answer a distinct scientific question.

---

## 28. Internet / literature search policy

You are authorized to browse the internet when:

```text
an implementation is unclear
a numerical issue blocks progress
recent latent-control / adapter / reversible-dynamics / flow work may solve the diagnosed failure
a library API changed
a paper contains directly relevant implementation detail
```

When browsing:

1. prefer primary sources: papers, arXiv, official repos, official docs;
2. save exact query, source URLs, paper/repo commit, and borrowed idea;
3. adapt methods to this frozen-latent problem rather than copying entire systems;
4. do not use literature to override project evidence.

Save `waveXX_external_research.md` whenever web research is used.

---

## 29. Problem-solving autonomy

If blocked by:

```text
shape mismatch
missing dependency
API incompatibility
numerical instability
non-finite gradients
ODE solver issue
memory limit
data-loader bug
checkpoint mismatch
```

diagnose and fix it yourself. You may inspect source, read official docs, search the web, write unit tests, change equivalent numerical implementations, reduce batch size, or use more stable formulations.

However, any change to:

```text
model architecture
loss
data
split
metric
claim threshold
```

after preregistration must move to the **next wave** rather than retroactively altering the current held-out experiment.

---

## 30. No p-hacking / no infinite same-test rescue

Autonomous iteration means:

```text
TRAIN/DEV: broad exploration allowed
HELD-OUT: one frozen evaluation per wave
```

After held-out, no same-wave rescue. Save failure, generate next-wave prompt, continue there.

Never erase negative results.

---

## 31. Artifact preservation per wave

Every Wave28/Wave29/Wave30/... must save at minimum:

```text
prompts/dynamics_waveXX.md
waveXX_preregistration.json
waveXX_frozen_manifest.json
waveXX_seed_manifest.json
waveXX_training_report.md
waveXX_development_results.md
waveXX_heldout_results.md
waveXX_ablation_results.md
waveXX_failure_taxonomy.md
waveXX_claim_decision.json
twenty_xx_wave_results.md
twenty_xx_wave_next_experiment.md
waveXX_external_research.md if browsing occurred
exact_commands.sh
environment_freeze.txt
files_changed.txt
tests_report.txt
updated_RESEARCH_LOG.md
updated_NEXT_EXPERIMENT.md
```

Do not overwrite previous wave files.

---

## 32. Research log discipline

Append an immutable log entry after every wave:

```text
date/time
git commit
prompt path
dataset hashes
frozen model hashes
models attempted
valid runs
invalid/discarded runs
development selection
held-out candidates
claim decisions
main failure mechanism
next-wave decision
```

Preserve invalid attempts with explanations.

---

## 33. Git discipline

For every completed wave:

1. run tests;
2. commit valid code/artifacts;
3. push according to repository policy;
4. record commit hash;
5. leave working tree clean;
6. never rewrite history to hide failed experiments.

---

## 34. Required Wave28 deliverables

Produce:

```text
prompts/dynamics_wave28.md
wave28_backbone_interface_audit.md
wave28_frozen_manifest.json
wave28_dataset_audit.md
wave28_preregistration.json
wave28_seed_manifest.json
wave28_q_dimension_results.md
wave28_language_encoding_results.md
wave28_subspace_results.md
wave28_field_dynamics_results.md
wave28_composition_results.md
wave28_loss_tournament.md
wave28_backbone_generality.md
wave28_no_switch_anchor_results.md
wave28_retarget_results.md
wave28_return_intent_results.md
wave28_return_latent_results.md
wave28_development_scorecard.csv
wave28_development_pareto.csv
wave28_final_candidate_selection.json
wave28_final_test_preregistration.json
wave28_heldout_results.md
wave28_ablation_table.csv
wave28_failure_taxonomy.md
wave28_claim_decision.json
twenty_eighth_wave_results.md
twenty_eighth_wave_next_experiment.md
wave28_external_research.md if browsing occurred
exact_commands.sh
environment_freeze.txt
files_changed.txt
tests_report.txt
updated_RESEARCH_LOG.md
updated_NEXT_EXPERIMENT.md
```

---

## 35. Wave28 claim decision JSON

Write `wave28_claim_decision.json` containing:

```text
C30_low_dim_adapter_preserves_frozen_behavior
C31_low_dim_intention_field_improves_retargeting
C32_dynamic_field_beats_static_residual
C33_learned_low_rank_subspace_beats_random_or_pca
C34_intention_return_symmetry_supported
C35_continuity_anchor_improve_editability
C36_adapter_generalizes_across_backbones
READY_FOR_CLOSED_LOOP_RETARGET
best_q_dimension
best_language_encoding
best_field_form
best_subspace_form
best_composition
best_loss_group
best_backbone
best_overall_adapter
random_subspace_control_result
pca_subspace_control_result
full_rank_control_result
no_switch_preserved
redirect_preserved
execution_redirect_preserved
continuity_improved
endpoint_identity_improved
decode_reencode_improved
return_intent_supported
return_latent_supported
adapter_parameter_count
adapter_fraction_of_total_parameters
next_wave_required
next_wave_number
next_wave_reason
```

Each scientific claim: `SUPPORTED / NOT_SUPPORTED / MIXED / NOT_TESTED`.

---

## 36. Required questions for every autonomous wave

Every report from Wave28 onward must answer:

1. What bottleneck from the previous wave was targeted?
2. Which encoding forms were tried?
3. Which q dimensions?
4. Which field dynamics?
5. Which projection/subspace forms?
6. Which composition operators?
7. Which loss combinations?
8. Which behavioral backbones?
9. Which controls were strongest?
10. Did no-switch behavior remain intact?
11. Did language retargeting improve?
12. Did execution-space retargeting improve?
13. Did continuity improve?
14. Did endpoint/decode-reencode identity improve?
15. Did low-rank match/beat full-rank control?
16. Did random/PCA bases work?
17. Did dynamic field beat static steering?
18. Did q-space show attractor-like convergence?
19. Did h0->h1->h0 reverse intention motion?
20. Did return require an explicit return flag?
21. Which ablation was most damaging?
22. Which loss term was necessary?
23. Which component was unnecessary?
24. Which parameterization was most efficient?
25. What does failure taxonomy say?
26. Was external research used?
27. What was borrowed from external sources?
28. Was held-out opened once after freeze?
29. Which claims are supported?
30. Is closed-loop retargeting justified?
31. If not, what is the clearest remaining bottleneck?
32. What next wave was generated and why?

---

## 37. Candidate final method wording

If supported, aim toward:

> **Language specifies an intention coordinate in a compact latent control field. A parameter-efficient residual adapter evolves this low-dimensional intention state and projects it into the frozen action latent, continuously retargeting the underlying policy without modifying its encoder or decoder. Because intention transitions are represented as relative motion in the same field, switching back to a previous instruction naturally reverses the intention displacement.**

Do not use this as a final claim until evidence supports it.

---

## 38. Candidate final paper story

Desired mature story:

```text
1. Actions form language-addressable, decodable coordinates.
2. Changing language causally changes future latent dynamics.
3. Full future trajectory prediction is brittle and unnecessarily difficult.
4. Expose a tiny intention-control field on top of frozen action coordinates.
5. Language changes the target/direction in this field.
6. The adapter continuously steers the frozen action latent with a low-rank residual.
7. No-switch behavior stays anchored to the original policy.
8. Mid-course instructions become online retargeting events.
9. Returning to a previous instruction is reverse movement in the same intention field.
10. Stored waypoints later provide physical memory for recoverable-state return.
```

Possible slogan:

> **Actions are coordinates; language moves the intention field.**

Alternative:

> **Language edits the future through a tiny latent force field.**

---

## 39. Do not overclaim

Do not claim until tested:

```text
language is a physical force
q potential is physical energy
robot motion is time reversible
return always restores the world
adapter solves arbitrary long-horizon planning
system is strictly stronger than all VLA/WAM methods
```

Use `latent force field`, `intention field`, `retargeting adapter`, and `recoverable-state return` carefully.

---

## 40. Autonomous stopping conditions

Continue Wave28 -> Wave29 -> Wave30 only while each new wave targets a scientifically motivated unresolved bottleneck.

### SUCCESS STOP

If a compact adapter satisfies S1–S7 on frozen held-out/prospective evaluation and no major ablation contradicts the mechanism, stop and generate:

```text
FINAL_METHOD_SUMMARY.md
FINAL_PAPER_STORY.md
FINAL_ABLATION_PLAN.md
NEXT_CLOSED_LOOP_RETARGETING_PROMPT.md
```

### REPRESENTATION STOP

If multiple well-motivated adapter/field/composition/loss families fail, including full-rank controls, and evidence shows the frozen action representation lacks required retargeting structure, stop adapter stacking and propose a temporally structured/contact-aware/state-action latent representation.

### DATA STOP

If evidence points to insufficient ordered retargeting data and no valid collector exists, generate `DATA_COLLECTION_REQUIREMENTS.md` with an exact collection specification.

### INFRASTRUCTURE STOP

If a required simulator/hardware capability is unavailable and no scientifically equivalent route exists, document the block precisely.

---

## 41. First actions

Start by:

1. reading the current project story;
2. reading Wave21–Wave27 reports and claim decisions;
3. auditing exact F1/F2/backbone interfaces;
4. auditing available ordered retarget data;
5. saving this exact prompt as `prompts/dynamics_wave28.md`;
6. writing `wave28_preregistration.json`;
7. creating the staged implementation matrix;
8. running sanity/gradient/leakage tests;
9. executing the Wave28 tournament;
10. freezing final held-out candidates;
11. evaluating held-out once;
12. generating reports and claim decisions;
13. if unresolved, automatically writing and executing Wave29 under the rules above.

Do not ask for confirmation unless credentials/permissions, a hard infrastructure limitation, or a safety constraint genuinely prevents progress.

Begin Wave28 now.
