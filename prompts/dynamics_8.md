# PGLT Twentieth-Wave Codex Prompt
# Prospective LIBERO Representation Motor-Margin Adjudication Before Dynamics

## 0. Mission

Wave 20 must adjudicate exactly one unresolved prerequisite from Wave 19:

> **Can the independent LIBERO action-coordinate representation preserve the very strong bidirectional language structure already observed, while achieving a genuine motor-fidelity safety margin rather than another threshold-edge result?**

Wave 19 already established that the LIBERO prospective data infrastructure is valid:

- official target resolved to `libero_10`;
- π0.5 source collection succeeded at scale;
- exact branch restoration succeeded;
- 297 certified source episodes were obtained;
- 415 eligible branches passed exact twin/source replay;
- the balanced primary dataset contains 240 episodes;
- the original split is frozen as 140 train / 50 development / 50 final test;
- the final 50-episode test split remains unopened.

Wave 19 representation semantics were very strong:

```text
action-to-text semantic delta = 0.940000
95% CI = [0.916667, 0.960000]

text-to-action semantic delta = 0.900000
95% CI = [0.866667, 0.933333]
```

Gripper fidelity passed.

The only failed prerequisite was continuous motor fidelity:

```text
correct-language continuous MSE = 0.00205037349
reconstruction-only continuous MSE = 0.00170801205

ratio = 1.200444393
frozen maximum = 1.200000000
```

The miss is numerically small but scientifically nonzero.

Therefore:

```text
Wave 20 is NOT a dynamics wave initially.
Wave 20 is NOT a F1/F2 rescue.
Wave 20 is NOT a seed sweep.
Wave 20 is NOT a hyperparameter search.
Wave 20 is a single-factor prospective representation adjudication.
```

If the new representation gate fails, stop.

Do not train F1/F2.

If it passes, freeze the representation and continue into the already-preregistered LIBERO F1/F2 pipeline exactly once.

---

## 1. Preserve Wave-19 Scientific Record

Do not reinterpret Wave 19.

Preserve:

```text
Wave19_snapshot_certification = PASS
Wave19_semantic_diagnostics = PASS
Wave19_gripper_fidelity = PASS
Wave19_continuous_motor_fidelity = FAIL
Wave19_representation_R_gate = FAIL

LIBERO_C3c_long = NOT_TESTED_REPRESENTATION_GATE_FAILURE
LIBERO_C4_closed_loop_refinement = NOT_TESTED_REPRESENTATION_GATE_FAILURE
LIBERO_C5_learned_direction_value = NOT_TESTED_REPRESENTATION_GATE_FAILURE
LIBERO_C6_proposal_recovery = NOT_TESTED_REPRESENTATION_GATE_FAILURE
```

Do not round `1.200444393` down.

Do not relabel Wave 19 as a pass.

Do not select a Wave-19 correct-language checkpoint.

The existing final test split must remain unread until explicitly authorized by the Wave-20 offline dynamics gate.

---

## 2. Wave-20 Scientific Hypothesis

The only new representation hypothesis is:

> **Increasing reconstruction pressure while preserving the same semantic factorization can retain language addressability and create a real motor-fidelity margin.**

The exact new correct-language objective is:

```text
L_total = 2.0 * L_reconstruction + 1.0 * L_semantic
```

This factor `2.0` is frozen before new confirmation data is read.

No alternative weight may be tested on the fresh confirmation-development set.

Do not test:

```text
1.25
1.5
1.75
2.5
3.0
```

unless Wave 20 is explicitly declared failed and a future wave is separately preregistered.

There is no sweep in Wave 20.

---

## 3. Fixed Representation Architecture

Keep exactly the same representation family as Wave 19:

```text
input:
action-only

action chunk horizon:
H_action = 16 LIBERO control steps

latent:
32-D total

semantic:
16-D

execution:
16-D

text encoder:
frozen OpenCLIP ViT-L/14 DataComp

semantic alignment:
same contrastive objective family

gradient isolation:
same Wave-19 implementation

training epochs:
40

EMA:
0.999

decoder:
same architecture

action normalization:
same frozen Wave-19 LIBERO statistics/protocol
```

No CALVIN representation/statistics/checkpoints may be loaded.

Do not alter architecture width/depth.

Do not alter latent dimensionality.

Do not add observation/state inputs.

Do not add contact inputs.

Do not add auxiliary losses.

Do not change optimizer family unless required to exactly reproduce Wave-19 configuration.

---

## 4. Paired Conditions Only

Train exactly two paired representation conditions.

### R0 — Reconstruction-only anchor

Use the exact Wave-19 reconstruction-only objective:

```text
L = L_reconstruction
```

### R1 — Motor-weighted correct-language model

Use:

```text
L = 2 * L_reconstruction + L_semantic
```

The pair must share:

- architecture;
- initialization protocol;
- seed list;
- train split;
- batching;
- optimizer;
- scheduler;
- number of epochs;
- EMA decay;
- decoder;
- action preprocessing.

Do not include shuffled-language as a trained primary paired model unless needed to compute the exact same semantic control metric already defined in Wave 19.

If shuffled-language must be trained for the semantic-delta control, keep it architecturally identical and do not use it for checkpoint selection.

---

## 5. Seeds

Register exactly six new seeds before any fresh confirmation episode is evaluated.

The seed list must not overlap Wave-19 representation seeds.

Write:

`wave20_seed_preregistration.json`

Required:

```text
6/6 seeds complete
all outputs finite
no seed dropped
no seed replaced
```

Do not run additional seeds after seeing results.

---

## 6. Preserve Existing Wave-19 Dataset Split

Keep unchanged:

```text
Wave-19 train split = 140 episodes
Wave-19 development split = 50 episodes
Wave-19 final test split = 50 episodes
```

Rules:

- use the 140 train episodes for Wave-20 representation training;
- the already-read 50 development episodes may be used for descriptive/dev diagnostics only;
- the existing 50 final test episodes must remain unopened;
- do not merge fresh confirmation-development data into training;
- do not modify source episode membership.

Record hashes of all three split manifests before Wave-20 training.

Write:

`wave20_existing_split_freeze.json`

---

## 7. Fresh Prospective Confirmation-Development Dataset

Wave 20 requires a new, genuinely fresh confirmation-development set.

Collect:

```text
5 new certified successful episodes per task
10 tasks
= 50 new certified episodes total
```

Use:

- official `libero_10`;
- the same fixed official π0.5 LIBERO checkpoint;
- a newly registered π0.5/environment seed schedule;
- the same exact-state snapshot protocol;
- the same future-support rules;
- the same action interface;
- the same control frequency;
- the same branch certification implementation.

These 50 episodes must not overlap any Wave-19 episode.

Do not replace old episodes.

Do not add these 50 episodes to the original final test.

They are a new:

```text
Wave20_fresh_confirmation_development_set
```

---

## 8. Fresh Collection Seed Freeze

Before collecting the new 50 episodes, write:

`wave20_collection_preregistration.json`

Include:

```text
π0.5 checkpoint hash/path
OpenPI commit
LIBERO commit
task list
attempt cap per task
environment seed schedule
policy seed schedule
required successes/task = 5
snapshot format
branch fractions
future-support rule
certification tolerances
action mutation safety rule
JIT mode
```

The JIT mode must match formal Wave-19 collection.

Do not disable Numba JIT as a cache workaround.

The prior mismatched JIT diagnostic is invalid and must not be repeated.

---

## 9. Snapshot Restore Implementation

Use the corrected restore order discovered in Wave 19:

```text
mj_setState
-> mj_forward
-> mj_setState
```

Rationale:

- `mj_forward` rebuilds qpos-dependent geometry/controller-query state;
- the second `mj_setState` restores the complete official integration payload including acceleration / warm-start fields.

Do not revert to:

```text
mj_setState -> mj_forward
```

because that previously introduced nonzero restore-boundary discrepancies.

Preserve action-copy safety:

```python
env.step(action.copy())
```

or equivalent immutable boundary.

---

## 10. Fresh Snapshot Certification

Every new confirmation episode must satisfy the same exact-state certification standard.

For every admitted branch:

- twin A restore;
- twin B restore;
- same source continuation replay;
- exact state comparison;
- controller comparison;
- object comparison;
- source predicate comparison;
- terminal-success agreement;
- all outputs finite.

Expected under corrected Wave-19 implementation:

```text
state discrepancy = 0
controller discrepancy = 0
object discrepancy = 0
predicate agreement = 100%
```

Do not weaken the standard.

If any nonzero mismatch appears:

1. diagnose implementation first;
2. preserve the raw episode;
3. do not change thresholds;
4. do not modify branch location based on result.

---

## 11. Confirmation Set Freeze

Only after exactly 5 certified episodes/task are available:

write:

`wave20_fresh_confirmation_manifest.json`

Include:

- source episode IDs;
- exact task labels;
- source hashes;
- branch hashes;
- snapshot hashes;
- seeds;
- success predicates;
- lengths;
- action counts.

Freeze this manifest before representation evaluation on these episodes.

---

## 12. Representation Training Protocol

Train R0 and R1 on the same frozen 140-episode Wave-19 training set.

No fresh Wave-20 confirmation episode may enter training.

For each seed:

```text
train R0
train R1
```

Use the exact same dataloader episode assignments.

If minibatch order is seeded, pair it by seed where practical.

Record:

- raw checkpoint;
- EMA checkpoint;
- train loss;
- reconstruction loss;
- semantic loss;
- per-dimension reconstruction MSE;
- gripper metric;
- text/action retrieval diagnostics.

Write:

`wave20_representation_training_report.md`

---

## 13. No Confirmation-Based Checkpoint Search

Do not evaluate multiple epochs on the fresh confirmation set and choose the best one.

The checkpoint rule must be frozen before fresh confirmation inference.

Preferred:

```text
EMA epoch 40
```

for all six seeds.

If Wave-19 used another exact checkpoint-selection rule, reproduce it.

Do not introduce early stopping on fresh confirmation data.

---

## 14. Wave-20 Representation Metrics

Evaluate on the **fresh 50-episode confirmation-development set**.

### 14.1 Semantic addressability

Compute:

```text
action-to-text macro R@1
text-to-action macro R@1
```

Define semantic delta exactly as Wave 19:

```text
semantic_delta =
correct-language
-
max(reconstruction-only, shuffled-language)
```

Report:

```text
mean delta
source-episode-clustered bootstrap 95% CI
per-task delta
per-seed delta
```

### 14.2 Continuous motor fidelity

Primary motor metric:

```text
continuous action MSE
```

Compare paired R1 vs R0:

```text
motor_ratio =
MSE_correct_language /
MSE_reconstruction_only
```

### 14.3 Gripper fidelity

Use the same sign-based gripper metric.

Report:

```text
accuracy_correct
accuracy_reconstruction
drop = reconstruction - correct
```

### 14.4 Secondary reconstruction diagnostics

Report:

```text
per-action-dimension MSE
translation MSE
rotation MSE
gripper logit/value error
action clipping/saturation rate
decoded action norm
```

These are descriptive unless explicitly in the gate.

---

## 15. Source-Episode Bootstrap

Use:

```text
10,000 replicates
cluster = source episode
task-stratified
seed = 200820
```

Do not bootstrap action chunks as independent units.

Write:

`wave20_representation_statistical_report.md`

---

## 16. New Stricter Representation Gate

Wave 20 must demonstrate a **real margin**, not another threshold-edge pass.

All conditions must pass on the fresh 50-episode confirmation-development set.

### Semantic gate

Require:

```text
mean action-to-text semantic delta > 0
mean text-to-action semantic delta > 0

clustered lower 95% > 0
for both directions
```

### Continuous motor gate

Require:

```text
motor_ratio <= 1.15
```

This is intentionally stricter than Wave 19's 1.20 threshold.

Do not relax it if the result is e.g. 1.151.

### Gripper gate

Require:

```text
gripper accuracy drop <= 0.02
```

This is intentionally stricter than Wave 19's 0.05 threshold.

### Completeness gate

Require:

```text
6/6 seeds complete
all outputs finite
no missing task
no missing confirmation episode
```

If ANY condition fails:

```text
Wave20_representation_gate = FAIL
```

and STOP.

Do not train F1/F2.

Do not open the old 50-episode final test.

Do not try another reconstruction weight.

Do not add more seeds.

Do not change latent dimensions.

Conclusion if failed:

> **The current factorized LIBERO representation family does not yet support a robust cross-domain semantic+motor replication under the preregistered motor-margin criterion.**

---

## 17. Seed-Level Robustness Diagnostic

Even if the aggregate gate passes, report:

```text
semantic delta per seed
motor ratio per seed
gripper drop per seed
```

Do not require every seed individually to satisfy the 1.15 ratio unless this is registered before inference.

However, if one seed is catastrophically unstable, report it.

Checkpoint selection must follow a pre-frozen rule.

Suggested selection rule:

```text
among seeds satisfying finite-output and positive bidirectional semantic delta,
select the seed with median continuous MSE ratio
```

or reuse the exact Wave-19 intended selection rule.

Freeze the rule before reading fresh confirmation outputs.

Write:

`wave20_checkpoint_selection_rule.json`

---

## 18. Representation Gate Decision

Write:

`wave20_representation_gate.json`

Required fields:

```text
semantic_A2T_mean_delta
semantic_A2T_lower95
semantic_T2A_mean_delta
semantic_T2A_lower95

correct_language_continuous_MSE
reconstruction_only_continuous_MSE
motor_ratio

correct_gripper_accuracy
reconstruction_gripper_accuracy
gripper_drop

all_six_seeds_complete
all_outputs_finite

gate_pass
```

No rounding before gate comparison.

Store full precision.

---

## 19. If Representation Gate Passes: Freeze Representation

Only if Section 16 passes:

1. select exactly one R1 checkpoint using the frozen rule;
2. freeze the checkpoint;
3. record SHA256;
4. freeze decoder;
5. freeze normalization;
6. freeze text encoder/version;
7. freeze representation manifest.

Write:

`wave20_frozen_libero_representation_manifest.json`

After this point:

```text
representation optimizer steps = 0
EMA updates = 0
decoder updates = 0
```

---

## 20. Then Train LIBERO F1/F2 Exactly Once

Only after representation authorization.

Train:

```text
F1 = free execution-latent predictor
F2 = exact matched iterative refinement
```

Use the exact Wave-19 preregistered dynamics design.

Do not redesign F1/F2.

Do not tune four refinement iterations.

Do not run DEL.

Do not add observation/state inputs.

Do not open final test yet.

Use only the already-frozen LIBERO train/development protocol.

No fresh confirmation episode may be merged into dynamics training.

---

## 21. Offline Dynamics Gate O1–O8

Run the already-specified LIBERO offline dynamics evaluation.

Evaluate:

```text
H1
H2
H4
H8
```

Primary metrics:

```text
execution latent MSE
decoded continuous MSE
execution kNN radius
local-PCA normal distance
normalized rollout AUC
correction-target cosine
positive correction fraction
iteration 0->4 curves
```

Required gate:

```text
O1:
F2 AUC < F1 AUC
with source-episode clustered upper 95% CI < 0

O2:
F2 H4 execution MSE < F1

O3:
F2 H8 execution MSE < F1

O4:
F2 H8 decoded continuous MSE < F1

O5:
F2 H8 execution kNN radius < F1

O6:
mean correction-target cosine > 0

O7:
positive correction fraction > 0.5

O8:
final F2 rollout mean empirical normal distance < F1
```

If O1–O8 fails according to the existing frozen decision rule:

STOP.

Do not open final test.

Do not retune F2.

---

## 22. Only Then Open the Untouched Wave-19 Final Test Split

The original 50-episode Wave-19 final test has remained unread.

It may be opened only if:

```text
Wave20 representation gate = PASS
AND
LIBERO offline O1–O8 authorization = PASS
```

Before opening, write:

`wave20_final_test_open_manifest.json`

Include:

- representation hash;
- F1 hash;
- F2 hash;
- decoder hash;
- normalization hash;
- exact test episode IDs/hashes;
- exact branch IDs/hashes;
- B0–B5 protocol;
- perturbation protocol;
- bootstrap seed;
- claim gates.

Once written, do not modify the models.

---

## 23. Final Closed-Loop Evaluation

Run from exact certified branch states:

```text
B0 = π0.5 source continuation reference
B1 = F1
B2 = F2
B3 = norm-matched random refinement
B4 = shuffled learned-direction refinement
B5 = negative refinement
```

All methods must start from identical physical snapshots.

No future action/state/task-success leakage.

F1/F2 must remain action-history latent continuation models.

Do not feed new RGB/state observation to them during continuation.

---

## 24. Proposal Perturbation Recovery

Only after final-test authorization.

Use frozen scales:

```text
sigma ∈ {0.05, 0.10, 0.20} × train execution std
```

Compare:

```text
F1_noisy
F2_from_noisy
random_from_noisy
negative_from_noisy
```

Do not tune sigma.

---

## 25. Closed-Loop Primary Endpoints

Primary:

```text
official LIBERO continuation success
```

Report:

```text
F2 - F1 absolute success difference
source-episode clustered 95% CI
per-task results
25% vs 50% branch results
```

Also report:

```text
decoded action error
physical q/TCP/object deviation
execution kNN radius
normal distance
time-to-success where valid
```

Do not treat expert imitation distance as identical to task success.

---

## 26. Direction-Specific Mechanism Controls

Support:

```text
LIBERO_C5_learned_direction_value
```

only if:

```text
F2 > norm-matched random
F2 > shuffled direction
F2 > negative refinement
```

and correction-target alignment remains positive.

If F2 > F1 but these controls fail:

```text
closed_loop_refinement_benefit = SUPPORTED
learned_direction_specificity = NOT_SUPPORTED
```

---

## 27. Cross-Domain Claim Logic

Possible successful Wave-20 outcome:

```text
CALVIN:
C1 supported
C2 supported
C3c-local strengthened
C3c-long supported
C3d supported

LIBERO:
representation semantic+motor gate passed
offline F2 > F1 at H1/H2/H4/H8
exact-state closed-loop F2 > F1
direction controls support learned correction
```

Then strongest paper story:

> **Across CALVIN and LIBERO, language supervision organizes continuous action latents into meaningful and executable coordinates. Free prediction advances these coordinates, while iterative refinement suppresses accumulated drift. On prospectively branchable LIBERO trajectories, this stabilization translates from latent-space improvements into better closed-loop continuation from identical physical states.**

If only representation passes but dynamics fails:

> **The semantic/executable action-coordinate representation replicates on LIBERO, but the CALVIN refinement advantage does not replicate under the frozen LIBERO dynamics protocol.**

If representation fails:

> **The current factorized representation family does not robustly reproduce the joint semantic-motor property on LIBERO under the preregistered margin.**

Do not hide a negative result.

---

## 28. Required Diagnostics for the Near-Threshold Representation Case

Because Wave 19 failed by only:

```text
0.000000759028 absolute continuous MSE
```

Wave 20 must explicitly determine whether the previous failure was:

```text
a genuine semantic-motor tradeoff
or
a threshold-edge optimization outcome
```

Report:

1. translation MSE ratio;
2. rotation MSE ratio;
3. per-dimension ratios;
4. seed-wise motor ratios;
5. raw vs EMA motor ratios;
6. correct-language vs reconstruction-only train MSE;
7. dev MSE;
8. fresh-confirmation MSE;
9. semantic delta vs motor ratio scatter;
10. whether increased reconstruction weight reduces motor penalty consistently across seeds.

This is analysis only.

Do not use these plots to change the gate.

---

## 29. Required Files

Produce:

```text
twentieth_wave_results.md
twentieth_wave_next_experiment.md

wave20_seed_preregistration.json
wave20_existing_split_freeze.json
wave20_collection_preregistration.json
wave20_fresh_confirmation_manifest.json

wave20_representation_training_report.md
wave20_representation_statistical_report.md
wave20_representation_gate.json
wave20_checkpoint_selection_rule.json

wave20_motor_margin_diagnostics.md

wave20_frozen_libero_representation_manifest.json
    only if gate passes

wave20_dynamics_preregistration.json
wave20_dynamics_results.md
    only if authorized

wave20_final_test_open_manifest.json
    only if authorized

wave20_closed_loop_results.md
wave20_intervention_results.md
wave20_perturbation_recovery.md
    only if authorized

wave20_cross_domain_claim_decision.json

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

## 30. Required Tests

Before fresh confirmation evaluation:

```text
Wave-19 train/dev/test manifest hashes unchanged
final 50 test episodes unread
fresh 50 confirmation episodes disjoint from all Wave-19 episodes
5 certified episodes per task
all snapshots exact
all predicate replays match
JIT mode equals formal collection
action-copy mutation guard passes
```

Representation tests:

```text
R0/R1 architecture identical
same six registered seeds
R1 loss weight exactly 2.0
EMA decay exactly 0.999
40 epochs
no extra hyperparameter sweep
no confirmation-based epoch selection
all outputs finite
```

Leakage tests:

```text
fresh confirmation episodes absent from training
final test absent from training/dev/confirmation
no future actions/states used by representation/dynamics
```

If downstream authorized:

```text
F2 exact F1 initialization
exact four refinement iterations
random norm matching
shuffled-source provenance
negative direction sign
source-episode clustered bootstrap
official LIBERO success predicate
```

Target:

```text
all tests pass
```

---

## 31. Exact Command Logging

Append every command to:

`exact_commands.sh`

Include:

```text
fresh π0.5 collection
snapshot certification
representation R0 training
representation R1 training
semantic evaluation
motor evaluation
bootstrap
gate decision
F1 training if authorized
F2 training if authorized
offline O1–O8 if authorized
final test opening if authorized
B0–B5 if authorized
perturbation recovery if authorized
tests
```

Do not rely on shell history.

---

## 32. Stop Conditions

STOP immediately if:

```text
fresh confirmation episodes are not independent

snapshot certification loses exactness

JIT mode differs from formal collection

Wave-19 final test is accidentally read early

R1 uses any reconstruction weight other than 2.0

additional seeds are run after seeing results

fresh confirmation data enters training

semantic lower 95% <= 0 in either direction

motor ratio > 1.15

gripper drop > 0.02

any seed missing/nonfinite
```

If representation gate fails:

```text
do not train F1/F2
do not open final test
do not try another loss weight
```

This stop is scientific, not operational.

---

## 33. Final Report Questions

The Wave-20 report must answer:

1. Were the Wave-19 140/50/50 splits preserved exactly?
2. Did the old 50-episode final test remain unopened until authorization?
3. Were 50 genuinely new certified confirmation episodes collected?
4. Did all fresh branch restores reproduce source continuations exactly?
5. Were six new seeds preregistered before evaluation?
6. Was the only scientific change `L = 2*L_rec + L_sem`?
7. Did bidirectional semantic deltas remain positive?
8. Were both clustered semantic lower 95% bounds > 0?
9. What was the new continuous MSE ratio?
10. Did it clear 1.15 with real margin?
11. What was the gripper accuracy drop?
12. Did all six seeds complete?
13. Was the representation gate PASS or FAIL?
14. Which action dimensions contributed most to motor penalty?
15. Did EMA improve the motor margin consistently?
16. Did increased reconstruction pressure reduce motor penalty across seeds?
17. If representation passed, what checkpoint was frozen and why?
18. If authorized, did F2 beat F1 offline at H1/H2/H4/H8?
19. What was the clustered AUC difference and CI?
20. If authorized, did the final held-out closed-loop F2 beat F1?
21. Did F2 beat random/shuffled/negative controls?
22. Did proposal perturbation recovery pass?
23. Which CALVIN phenomena replicated independently on LIBERO?
24. What exact cross-domain paper story is now defensible?
25. What remains before submission?

---

## 34. Strategic Meaning

Wave 19 already succeeded at the hard infrastructure part:

```text
π0.5 source generation
+
official LIBERO-10
+
exact MuJoCo branch states
+
deterministic restore
+
large certified branchable dataset
```

The experiment did not reach dynamics because the representation missed the continuous-motor gate by a tiny but valid amount.

Wave 20 should therefore answer one focused question with one controlled intervention:

```text
Can stronger reconstruction pressure preserve the semantic structure
while creating a genuine motor-fidelity margin?
```

If yes:

```text
freeze representation
-> train F1/F2 once
-> run O1–O8
-> only then open untouched final test
-> run exact-state B0–B5
```

If no:

```text
stop the current representation family
```

Do not let a downstream refinement result rescue a failed representation prerequisite.

The point of Wave 20 is not to make the paper positive.

The point is to determine whether the cross-domain story survives a stricter prospective test.
