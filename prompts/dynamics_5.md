# PGLT Seventeenth-Wave Codex Prompt — Continuous-Play Long-Horizon Refinement Across Language Boundaries

## Purpose

This wave amends the previous operational rule that a primary dynamics
trajectory must fit inside one atomic language annotation.

Wave 16 showed that this rule makes H4/H8 unavailable in the audited public
CALVIN annotations. It did **not** establish that crossing annotation
boundaries is automatically valid. Wave 17 must prove physical/session
continuity independently and explicitly separate boundary-crossing rollouts
from naturally boundary-free rollouts.

CALVIN atomic language segments are often short, while the underlying play stream is physically continuous across successive annotated skills. The dynamics question concerns continuous motion evolution:

q_t -> q_{t+1} -> ...

Therefore the correct long-horizon evaluation unit is a **continuous physical play block**, not a single atomic language segment.

Wave 16 / public-data replication already established an independent short-horizon result on VyoJ CALVIN data:

```text
60 external segments
6 tasks × 10 segments/task
64–65 frames per segment
4 non-overlapping H=16 windows per segment
H1/H2 evaluated only

F1 AUC = 0.728379
F2 AUC = 0.632797
Delta_AUC = F2 - F1 ≈ -0.095582
95% CI = [-0.113417, -0.079462]
all six task-specific bootstrap upper 95% bounds < 0
external replication gate = PASS

H1 execution MSE:
F2 0.670658 < F1 0.749443

H2 execution MSE:
F2 0.655076 < F1 0.776540

H1/H2 decoded continuous error:
F2 < F1

H1/H2 execution kNN radius:
F2 < F1

mean correction-target cosine = 0.415405
fraction correction cosine > 0 = 94.4%

across all four frozen refinement iterations:
latent error decreases
decoded error decreases
execution kNN radius decreases
```

Therefore:

```text
C3c-local =
STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION

C3c-long =
NOT_TESTED
```

Do not extrapolate H1/H2 to H4/H8.

The purpose of wave 17 is to test H4/H8 on continuous CALVIN play trajectories.

## 1. Frozen Scientific State

Preserve:

```text
C1 language-addressable continuous action coordinates = SUPPORTED
C2 executable continuous action coordinates = SUPPORTED
C3a full-latent DEL = REJECTED
C3b executable-subspace decoder-grounded DEL = REJECTED
C3c-local generic matched refinement =
STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION
C3c-long = NOT_TESTED
```

No DEL rescue.

No representation retraining.

No F1/F2 retraining.

Use the exact frozen:

- representation checkpoint;
- shared semantic predictor;
- F1 execution MLP;
- F2 matched refinement;
- F2 objective;
- F2 initialization;
- F2 iteration count;
- normalization;
- decoder;
- language/text encoder.

Required:

```text
representation optimizer steps = 0
F1 optimizer steps = 0
F2 optimizer steps = 0
EMA updates = 0
```

## 2. Revised Long-Trajectory Unit

For wave-17 eligibility, supersede—but do not erase from the historical
wave-16 record—the old rule:

```text
one long trajectory must remain inside one atomic language annotation
```

Replace it with:

```text
one long trajectory block must remain inside one physically continuous CALVIN play stream
```

A valid block MAY contain multiple atomic language/task annotations.

Primary validity concerns:

- physical continuity;
- action continuity;
- frame continuity;
- no simulator/environment reset;
- no episode/session reset;
- no missing frames;
- no duplicated frames;
- no synthetic concatenation.

Language boundaries are metadata, not mandatory truncation points, only after
the physical-continuity gate has passed.

## 3. Primary Block-Length Requirement

The latent window remains:

```text
H_action = 16 raw CALVIN frames
stride = 16
```

To support a causal two-state input plus an H8 rollout, require:

```text
>= 10 consecutive non-overlapping H16 windows
```

which corresponds to:

```text
>= 160 contiguous source frames
```

IMPORTANT:

The 160-frame requirement applies to the **continuous physical play block**.

It does NOT apply to one atomic task annotation.

A 160+ frame block may span multiple language annotations.

Preferred:

```text
>= 12 H16 windows
```

when available, to support multiple H8 rollout starts.

Do not manufacture length by stitching disconnected demonstrations.

## 4. Public-Data-First Source

Use the open CALVIN source already validated in wave 16 where possible.

Primary candidate source:

```text
VyoJ/calvin-ABCD-D-subsets
```

Reuse already-downloaded source files and hashes.

Do not redownload files unnecessarily.

The current 60 task-centered 64–65-frame segments may be used as anchors to locate their source frames inside the original continuous play stream.

Do not concatenate the 60 extracted segments.

Recover actual surrounding frames from the same continuous physical stream.

## 5. Construct Continuous Play Blocks

For every candidate anchor or play location:

1. identify the original source shard;
2. identify the exact global/source frame IDs;
3. map the anchor to exactly one authoritative `ep_start_end_ids.npy` row and
   treat that row as the candidate source-session boundary;
4. expand backward/forward only inside that same continuous session;
5. stop at any reset/discontinuity;
6. construct maximal contiguous blocks;
7. split maximal blocks deterministically into eligible evaluation blocks.

Record:

```text
source_repo
source_shard
source_environment
source_session_id
start_frame
end_frame
frame_count
number_H16_windows
annotation_boundaries
annotation_sequence
task labels
language strings
reset flags
```

Do not inspect F1/F2 predictions during block construction.

Contiguous numeric frame IDs alone do not prove a common session. Never cross
an `ep_start_end_ids.npy` row boundary, even if the adjacent global frame IDs
are consecutive and the kinematic jump happens to be small.

## 6. Physical Continuity Audit

A block is valid only if physical continuity is proven.

Use official/source metadata first.

Check, where available:

- episode/session IDs;
- reset markers;
- scene metadata;
- contiguous raw frame IDs;
- timestamp continuity;
- robot state continuity.

Add a secondary kinematic sanity check using robot observations.

For consecutive raw frames report:

- arm joint delta norm;
- TCP position jump;
- TCP orientation jump;
- gripper jump.

Thresholds for discontinuity must be frozen using source/training statistics before F1/F2 evaluation.

Do not remove difficult but physically valid motion because it has large model error.

If official metadata says reset, reject regardless of kinematic smoothness.

Save:

`continuous_play_integrity_audit.json`

## 7. Language Annotation Boundaries as Metadata

For each H16 window record:

```text
annotation_id
canonical_task_if_any
language_if_any
boundary_inside_window
boundary_before_next_window
```

Do not require every frame to have a language annotation.

Unlabeled intervals must remain:

```text
NO_LANGUAGE_ANNOTATION
```

Do not invent labels.

The frozen models have no preregistered null-language input. Therefore a
Protocol-A primary rollout start must have a causally available annotation at
its issue frame. Starts with no active annotation may be reported as data
availability diagnostics, but must not be assigned a newly invented null text
embedding or enter the primary F1/F2 comparison.

## 8. Critical Context-Leakage Problem

F1/F2 use semantic/context inputs.

Crossing a task boundary creates a key issue:

**the future annotation schedule cannot silently be given to an autonomous rollout.**

Therefore wave 17 MUST separate two evaluation protocols.

## 9. Protocol A — Causal Context-Held Rollout

This is the strict autonomous primary protocol.

At rollout start, use only the context causally known at the start window.

During H1/H2/H4/H8 rollout:

```text
do not inject future annotation/task labels
do not switch context using future ground-truth boundaries
```

If the frozen implementation has no online context-update mechanism, use:

```text
context = start-context held fixed
```

for the entire rollout.

The start context is the annotation active at the first frame of the current
H16 window. If an annotation boundary lies inside that window, retain the
start-of-window context and record the within-window boundary; do not look
ahead to choose the next label.

This protocol measures:

```text
autonomous latent dynamics under no future task-plan information
```

Absolute errors may grow when the source teleoperator changes goals.

That is expected and must be discussed.

The F1 vs F2 paired comparison remains valid.

## 10. Protocol B — Exogenous Ground-Truth Context Schedule

This is a secondary controlled-dynamics diagnostic.

Supply the true CALVIN annotation/task context when the source play stream changes annotation.

Both F1 and F2 receive the exact same context schedule.

Freeze this scheduling rule before inference:

```text
context for a latent window = annotation active at that window's first frame
boundary inside an H16 window = context changes no earlier than the next H16 window
unlabeled window = ineligible unless an already-frozen null context exists
```

Do not compare alternative boundary-assignment rules after seeing F1/F2
outputs.

Label this:

```text
EXOGENOUS_CONTEXT_SCHEDULE_DIAGNOSTIC
```

It is NOT fully autonomous task planning.

It asks:

> If the high-level task/context schedule is externally supplied, does F2 improve low-level latent rollout stability across long continuous play?

Do not mix Protocol B into Protocol A primary claims.

## 11. Optional Protocol C — Boundary-Free Subset

As a diagnostic only, identify rollout starts whose H4 or H8 horizon does not cross a language annotation boundary.

Do not require the whole 160-frame block to be one task.

Report F1/F2 on this naturally occurring subset if sample count is sufficient.

This separates:

```text
pure long-horizon dynamics error
from
high-level task-switch ambiguity
```

Do not use this subset as the sole primary benchmark.

## 12. Block Sampling and Leakage Prevention

Construct blocks before F1/F2 inference.

Avoid heavy duplication.

Preferred:

```text
no raw-frame overlap between primary evaluation blocks
```

Do not count many overlapping crops from the same play segment as independent trajectories.

The inferential bootstrap unit must be the source play session, not individual
windows or multiple blocks cut from the same session.

## 13. Data Adequacy Gate

Remove the old requirement:

```text
10 long blocks per atomic task
```

New primary target:

```text
>= 60 non-overlapping continuous play blocks
>= 10 H16 windows/block
>= 10 distinct source play sessions
```

Also require dataset-level rollout starts:

```text
H1 >= 300 starts
H2 >= 250 starts
H4 >= 150 starts
H8 >= 60 starts
```

Do not force task balance.

Instead report:

- task composition;
- annotation transition composition;
- unlabeled fraction;
- number of boundary crossings.

If open data cannot satisfy this, mark the confirmatory H8 gate underpowered.

Do not silently weaken the gate.

## 14. Prospective Manifest

Before H4/H8 inference, write:

`wave17_continuous_play_preregistration.json`

Include:

- exact source files;
- SHA256;
- exact selected blocks;
- raw frame ranges;
- session IDs;
- no-reset evidence;
- annotation sequence;
- H16 window indices;
- valid H1/H2/H4/H8 starts;
- Protocol A context rule;
- Protocol B context rule;
- frozen model hashes;
- metrics;
- bootstrap method;
- claim gates;
- explicit statement that H4/H8 outputs have not been read.

Only then may H4/H8 inference begin.

Also freeze a source-session-to-block table. Blocks from the same source
session remain statistically clustered even when their raw frame ranges do not
overlap.

## 15. Preserve Prior H1/H2 Replication Correctly

The previous 60-segment H1/H2 VyoJ result is historical external replication.

Preserve it exactly.

Verify these immutable wave-16 baselines before wave-17 acquisition or
inference:

```text
artifacts/sixteenth_wave/external_h12/selected_segments_manifest.json
SHA256 309d51dc3000bb558b59b7ecf4678104afa51e86fee013493cbb326d56cebcbc

results/dynamics/sixteenth_wave/2026-08-13_dynamics_4_external_h12/external_h12_prospective_preregistration.json
SHA256 54a6161f61759d8c7a9d02c6bd08ad8ec35779520464264c2fd5e4c87d62d54d

results/dynamics/sixteenth_wave/2026-08-13_dynamics_4_external_h12/external_h12_paired_trajectory_bootstrap.json
SHA256 daead5c7bf837d987089bd82644d44a1cb8f88a9e33afcb3a6179cb5d9165817

results/dynamics/sixteenth_wave/2026-08-13_dynamics_4_external_h12/wave16_external_h12_claim_decision.json
SHA256 25a5c68e9ca040bef90ada60cb6cfe632a828a16805f50efd4dfb31c62eacf6b
```

If any hash differs, stop and report the mismatch. Do not regenerate a
historical artifact to make the check pass.

If some source frames overlap wave-17 blocks:

- report overlap;
- do not call within-wave17 H1/H2 another independent replication.

Wave 17's novel confirmatory evidence is H4/H8.

## 16. Autonomous Rollout

For both F1 and F2:

- start from the same true causal initial latent history;
- after rollout begins, feed predicted latent states back recursively;
- do not teacher-force intermediate true latents;
- do not use future raw actions;
- do not use future robot states.

F2:

```text
e_init = F1(...)
e_refined = frozen_F2_refinement(e_init, ...)
```

Keep the exact frozen four-step refinement.

## 17. Primary Horizons

Evaluate exactly:

```text
1
2
4
8
```

latent steps.

At original CALVIN 30 Hz and H16:

```text
1 step ≈ 0.5333 s
2 steps ≈ 1.0667 s
4 steps ≈ 2.1333 s
8 steps ≈ 4.2667 s
```

Verify source frequency before reporting.

No padding.

## 18. Metrics

For every protocol and horizon report:

### Execution latent
- MSE;
- normalized MSE;
- cosine similarity.

### Full latent
- full MSE;
- semantic MSE;
- execution MSE.

### Decoded actions
- continuous action MSE;
- gripper accuracy;
- per-dimension error.

### Off-manifold
- execution nearest-training-latent distance;
- execution kNN radius;
- full-latent kNN radius;
- fraction beyond frozen train threshold.

### Refinement mechanism
- correction-target cosine;
- fraction cosine > 0;
- per-refinement-iteration latent error;
- per-iteration decoded error;
- per-iteration kNN radius.

## 19. Primary Statistical Endpoint

Primary protocol:

```text
Protocol A — causal context-held
```

Primary endpoint:

paired physical-block-level normalized rollout-error AUC over:

```text
H1/H2/H4/H8
```

For block i:

```text
Delta_AUC_i = AUC_F2_i - AUC_F1_i
```

Use:

```text
first average paired block Delta_AUC values within each source session
paired source-session bootstrap
10,000 replicates
95% CI
```

If and only if every selected block comes from a distinct source session, this
reduces to a paired whole-block bootstrap. Otherwise, blocks must not be
treated as independent resampling units.

Primary long-horizon gate requires:

```text
upper_95_CI(Delta_AUC) < 0
```

Do not bootstrap windows or correlated blocks as independent samples. Report
the block-level distribution descriptively in addition to the session-level
inferential endpoint.

## 20. H4/H8 Hard Gate

For Protocol A require:

```text
F2 H4 execution MSE < F1 H4 execution MSE
F2 H8 execution MSE < F1 H8 execution MSE
F2 H8 decoded continuous MSE < F1 H8 decoded continuous MSE
F2 H8 execution kNN radius < F1 H8 execution kNN radius
```

Also require H8 sample-count adequacy.

If H8 has too few valid starts:

```text
C3c-long = NOT_TESTED_INSUFFICIENT_H8_SUPPORT
```

Do not infer from H4 alone.

## 21. Boundary-Stratified Analysis

Classify every rollout start by:

```text
0 boundaries crossed
1 boundary crossed
2+ boundaries crossed
```

Also record:

```text
same annotated task
task changes
labeled->unlabeled
unlabeled->labeled
```

Report F1/F2 differences separately.

Key question:

> Does refinement remain beneficial when the physical trajectory crosses semantic/task boundaries?

## 22. Context-Sensitivity Analysis

Compare Protocol A vs Protocol B.

Questions:

1. How much absolute error comes from missing future high-level context?
2. Does F2 beat F1 under both protocols?
3. Does F2 advantage change near boundaries?
4. Does exogenous context improve both models similarly?

If F2 only wins under Protocol B:

```text
generic refinement benefit is conditional on externally supplied task schedule
```

Do not claim fully autonomous long-horizon stability.

Set `context_dependency = ROBUST_TO_BOUNDARIES` only if Protocol A passes its
primary gate and the boundary-crossing subset contains at least 30 H8 starts
from at least 10 source sessions with a session-clustered Delta_AUC upper 95%
bound below zero. If Protocol A fails but Protocol B passes, set
`BENEFIT_REQUIRES_EXOGENOUS_CONTEXT`. If boundary support is smaller than the
stated minimum, set `UNRESOLVED`; do not convert a small descriptive subset
into a robustness claim.

## 23. Refinement Mechanism Gate

For H4/H8 require:

1. mean correction-target cosine > 0;
2. fraction positive correction cosine > 0.5;
3. final refinement lowers execution kNN radius relative to F1;
4. final refinement lowers decoded-action error relative to F1;
5. refinement iteration curve shows net improvement from iteration 0 to K.
6. mean empirical normal distance decreases from F1 to F2;
7. normal-distance reduction has positive association with decoded-action
   improvement.

Do not require strict monotonicity at every individual sample.

## 24. Empirical Manifold Audit

Reuse the frozen dynamics-training execution-latent reference.

Do not fit manifold statistics on wave-17 evaluation data.

Use the existing kNN/local-PCA procedure.

Report:

- F1 empirical normal distance;
- F2 empirical normal distance;
- normal-distance reduction;
- relation to decoded-action improvement.

Call it an empirical executable latent manifold, not a physical manifold.

## 25. Optional q-Space Diagnostic

If exact robot joint state is available for every frame:

- record q-space displacement over each H16 window;
- record cumulative joint-space path length;
- report whether latent prediction error correlates with physical-motion magnitude or annotation boundaries.

q-space is evaluation-only.

Do not train on it in this wave.

## 26. Historical DEL Baseline

No DEL rescue.

If the frozen historical DEL can safely run on continuous blocks, report it separately.

Otherwise:

```text
DEL = historical negative baseline only
```

No training or tuning.

## 27. Claim Decision

Write:

`wave17_claim_decision.json`

Maintain:

```text
C1 = SUPPORTED
C2 = SUPPORTED
C3a_full_DEL = REJECTED
C3b_exec_DEL = REJECTED
C3c_local =
STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION
```

Set:

```text
C3c_long =
SUPPORTED
or
REJECTED
or
NOT_TESTED_INSUFFICIENT_H8_SUPPORT
```

Set independently:

```text
C3d_refinement_manifold_stabilization =
SUPPORTED
or
NOT_SUPPORTED
```

Also set:

```text
context_dependency =
ROBUST_TO_BOUNDARIES
or
BENEFIT_REQUIRES_EXOGENOUS_CONTEXT
or
UNRESOLVED
```

## 28. Paper Story Rules

### If C3c-long passes under Protocol A

Use:

> **Language grounds meaningful action coordinates; prediction advances the latent trajectory, while iterative refinement suppresses accumulated drift across continuous robot motion, including semantic task boundaries.**

Short:

> **Language anchors action meaning; refinement stabilizes continuous latent evolution.**

Chinese:

> **语言为动作 latent 提供语义锚点；自由预测负责推进，而迭代 refinement 在连续机器人运动中抑制累积漂移，即使轨迹跨越多个原子任务边界。**

### If C3c-long passes only under Protocol B

Use:

> **Given an exogenous high-level task schedule, matched refinement improves long-horizon latent dynamics across continuous robot motion.**

Do not call it autonomous planning.

### If H4 improves but H8 fails

Use:

> **Refinement extends the stable prediction horizon beyond local transitions, but evidence for 4+ second autonomous latent rollout remains insufficient.**

### If C3c-long fails

Use:

> **Language-grounded action coordinates are semantically addressable, executable, and locally predictable; matched refinement improves local transitions, while stable long-horizon dynamics remains unresolved.**

## 29. Integrity Tests

Add tests for:

- source frame continuity;
- same physical session;
- no reset crossing;
- no synthetic concatenation;
- H16/stride16;
- >=10 windows/block;
- H8 valid starts;
- annotation boundaries allowed;
- boundary metadata correct;
- future annotations excluded from Protocol A;
- Protocol B labeled exogenous;
- no future raw actions;
- no teacher forcing;
- frozen F1/F2 hashes;
- no optimizer/backward;
- prior H1/H2 replication artifact immutable;
- source-session-clustered paired bootstrap;
- no overlapping blocks counted as independent;
- manifold reference training-only;
- all metrics finite.

Do not overwrite prior artifacts.

F2's frozen inference-time `autograd.grad` refinement is required and is not a
training backward call. Tests must forbid loss `.backward()`, optimizer
construction/steps, checkpoint mutation, and EMA updates while permitting the
exact frozen four-step refinement computation.

## 30. Disk Safety and Staged Source Handling

Before each download run both:

```bash
df -h /home/jinjaguo/PGLT
df -B1 /home/jinjaguo/PGLT
```

Update a wave-17 `disk_budget.json`. Never intentionally reduce available
space below 200 GB. Reuse already audited `_023` and `_000` hashes/files where
present. For every additional VyoJ subset: download one shard, hash it,
extract or stream only required metadata/actions/robot observations, retain
only eligible compact blocks, delete the ZIP and unnecessary temporary files,
then recheck disk. Do not keep a complete ZIP and complete extraction
simultaneously.

## 31. Required Deliverables

Produce:

- `seventeenth_wave_results.md`
- `seventeenth_wave_next_experiment.md`
- continuous-play source audit
- source-session reconstruction report
- physical-continuity audit
- continuous-block manifest
- annotation-boundary metadata
- H1/H2/H4/H8 support table
- prospective H4/H8 preregistration
- frozen model hash manifest
- Protocol-A results
- Protocol-B results
- optional Protocol-C results
- source-session-clustered paired bootstrap
- boundary-stratified analysis
- context-sensitivity analysis
- horizon-wise latent metrics
- decoded-action metrics
- off-manifold metrics
- refinement iteration curves
- correction-target alignment
- empirical manifold analysis
- optional q-space diagnostic
- frozen DEL negative-baseline note
- final claim decision JSON
- exact commands
- environment/provenance
- files-changed report
- full tests
- updated `RESEARCH_LOG.md`
- updated `NEXT_EXPERIMENT.md`

Final report must answer explicitly:

1. How many physically continuous play sessions were reconstructed?
2. How many >=10-window continuous blocks were eligible?
3. How many blocks cross 0/1/2+ annotation boundaries?
4. How many H1/H2/H4/H8 rollout starts exist?
5. Did any primary block cross a reset or discontinuity?
6. Was block construction frozen before H4/H8 inference?
7. Were F1/F2 completely frozen?
8. Does F2 beat F1 on Protocol-A source-session-level AUC with upper 95% CI < 0?
9. Does F2 beat F1 at H4?
10. Does F2 beat F1 at H8?
11. Does F2 reduce H8 decoded-action error?
12. Does F2 reduce H8 execution kNN radius?
13. Does F2 advantage persist across annotation boundaries?
14. Does F2 remain beneficial without future task labels?
15. How much does Protocol B exogenous context improve F1 and F2?
16. Is correction-target cosine still positive at H4/H8?
17. Do refinement iterations still reduce error and manifold distance at H4/H8?
18. Is C3c-long supported?
19. Is C3d supported?
20. Is the benefit robust to semantic task boundaries?
21. Does DEL remain permanently a negative baseline?
22. What final paper story is scientifically defensible?
23. Is any additional data collection still needed?

Do not write the desired conclusion before H4/H8 evaluation.

Core principle:

**continuous physical trajectory defines the dynamics unit; language annotations describe semantic events along that trajectory but do not artificially truncate the dynamics.**
