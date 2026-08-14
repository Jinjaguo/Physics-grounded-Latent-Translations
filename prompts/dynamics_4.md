# PGLT Sixteenth-Wave Codex Prompt — Open-Data-First Long-Horizon Refinement Evaluation

## Post-audit amendment — prospective public-data H1/H2 external replication

This section was added on 2026-08-13 **after** the initial open-data audit
established that the public CALVIN task annotations are at most approximately
65 frames long.  It is a user-requested modification of the original wave-16
long-horizon plan below, not part of the original preregistration.

For this amended experiment, the following rules supersede the incompatible
`>=160 frames`, `10 H16 windows`, and `H1/H2/H4/H8` requirements below:

```text
evaluation role = prospective external replication of wave-15 C3c-local
public source = VyoJ/calvin-ABCD-D-subsets
tasks = the frozen six CALVIN tasks
direct segment length >= 64 contiguous annotation-consistent frames
latent window H = 16 frames
primary stride = 16 frames
windows per selected segment >= 4
evaluated rollout horizons = H1 and H2 only
H4 and H8 = not run in this amended experiment
target = 10 segments/task, 60 total
```

Selection must remain model-independent.  Process subsets in the already
registered staged order, beginning with `subset_training_023`, then `_000`,
`_001`, and so on only until 10 direct eligible segments have been retained
for every task.  Select by exact task metadata, annotation length, continuity,
source frame availability, and original 30-Hz 7-D CALVIN `rel_actions` only.
Do not inspect F1/F2 outputs until the exact selected segments and their source
hashes have been frozen in a prospective manifest.

Use exactly four non-overlapping H16 windows from each selected segment.  A
65th leftover frame is not padded, repeated, or added to a window.  With four
latent windows, valid autonomous starts are:

```text
H1 starts per segment = 2
H2 starts per segment = 1
```

The representation, shared semantic predictor, F1 execution MLP, F2 matched
refinement model, normalization, decoder, F2 objective, F2 initialization,
and four refinement iterations remain exactly frozen from wave 15.  Required:

```text
representation optimizer/backward steps = 0
F1 optimizer/backward steps = 0
F2 optimizer/backward steps = 0
EMA updates = 0
future target actions = unavailable to both F1 and F2
```

Primary external-replication endpoint:

```text
per-trajectory normalized execution-error AUC over H1/H2
Delta_AUC = AUC_F2 - AUC_F1
paired whole-trajectory bootstrap = 10,000 replicates
replication success requires upper_95_CI(Delta_AUC) < 0
```

Also report H1/H2 decoded continuous-action error, gripper accuracy,
execution/full-latent off-manifold distances against the frozen wave-15
training reference, refinement correction-target cosine, and every F2
intermediate refinement state.  Report pooled, per-task, and per-source
results.  Do not reopen or retrain DEL.

The final wave-16 report and `RESEARCH_LOG.md` must explicitly state that the
amended external replication evaluated **H1 and H2 only** and did not evaluate
H4/H8.  Do not write a result conclusion until acquisition, preregistration,
frozen inference, and statistical analysis are complete.

## Purpose

This is a complete rewrite of wave 16.

Do **not** assume new CALVIN VR collection is required.

The first objective is to determine whether enough compatible long trajectories can be obtained from existing local data and open public CALVIN-derived datasets. Only if the open-data pipeline fails should manual/VR collection remain as a fallback.

The representation and dynamics models remain frozen.

Scientific state:

```text
C1 language-addressable action coordinates = SUPPORTED
C2 executable continuous action coordinates = SUPPORTED
C3a full-latent DEL = REJECTED
C3b executable decoder-grounded DEL = REJECTED
C3c-local matched generic refinement = SUPPORTED
```

The primary long-horizon question is:

> Does the already-frozen F2 matched-refinement model improve H=1/2/4/8 autonomous rollout stability over the already-frozen F1 execution MLP on genuinely long, annotation-consistent, action-compatible trajectories?

The secondary mechanism question is:

> Does F2 act as a corrective operator that reduces prediction error and empirical executable-manifold drift over long horizons?

DEL is frozen as a negative baseline. Do not attempt another DEL rescue.

---

# 0. Hard Constraints

Do not change:

- the wave-11 epoch-40 EMA representation;
- the 32-D latent;
- the 16-D `z_sem` / 16-D `z_exec` factorization;
- representation normalization;
- frozen decoder;
- frozen text encoder/projection;
- the wave-15 shared semantic predictor;
- F1 execution MLP checkpoint;
- F2 matched-refinement checkpoint;
- F2 objective;
- F2 refinement iteration count;
- F2 initialization from F1;
- action-window H=16;
- primary stride=16.

Required:

```text
representation optimizer steps = 0
F1 optimizer steps = 0
F2 optimizer steps = 0
EMA updates = 0
```

No future target actions may enter F1/F2.

Do not reopen DEL.

---

# 1. Disk-Safety Policy for the 1-TB Drive

Before any download:

```bash
df -h
du -sh <project_root> <existing_calvin_data_roots> 2>/dev/null || true
```

Create:

`artifacts/sixteenth_wave/data_acquisition/disk_budget.json`

Record:

- filesystem;
- total capacity;
- used bytes;
- free bytes;
- local CALVIN bytes;
- planned download;
- planned temporary extraction overhead.

Hard rule:

```text
never intentionally reduce free space below 200 GB
```

unless the user explicitly overrides it later.

Do not download the full ~700-GB ABCD Hugging Face mirror in one shot.

Use staged download, audit, candidate extraction, and cleanup.

For each temporary subset:

1. download;
2. hash;
3. extract;
4. inspect;
5. copy only eligible candidate metadata/action files to a permanent candidate cache;
6. delete the temporary ZIP and unnecessary extracted files;
7. verify free disk again.

Never keep both every large ZIP and every fully extracted copy.

---

# 2. Source Priority

Use this source order.

## Tier 0 — Existing Local CALVIN Data

Audit all existing local CALVIN directories first.

Search for:

- `task_D_D`
- `task_ABC_D`
- `task_ABCD_D`
- `auto_lang_ann.npy`
- `episode_*.npz`
- existing LeRobot conversions
- existing RoboVerse CALVIN trajectory files

Do not redownload files already present.

Create:

`local_calvin_inventory.json`

with paths, sizes, frame/action format, annotation metadata, and hashes where practical.

---

## Tier 1A — Small CALVIN D/D LeRobot Mirror for Metadata Scouting

Public source:

```text
CollisionCode/calvin_d_d_lerobot_v2.1
```

This mirror is small enough for scouting.

First download metadata only.

Preferred command if `hf` CLI is installed:

```bash
hf download CollisionCode/calvin_d_d_lerobot_v2.1 \
  --repo-type dataset \
  --include "meta/*" \
  --local-dir <DATA_ROOT>/hf_calvin_d_d_lerobot_meta
```

Fallback:

```bash
huggingface-cli download CollisionCode/calvin_d_d_lerobot_v2.1 \
  --repo-type dataset \
  --include "meta/*" \
  --local-dir <DATA_ROOT>/hf_calvin_d_d_lerobot_meta
```

Or use `huggingface_hub.snapshot_download`.

Audit:

- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `meta/info.json`

Extract for every episode:

```text
episode_index
length
task descriptions
task IDs
```

Important compatibility warning:

This LeRobot conversion may use a different frame rate / converted time base from the frozen 30-Hz CALVIN action representation.

Therefore:

```text
DO NOT use this mirror as primary F1/F2 evaluation data
unless exact temporal/action compatibility is proven.
```

Use it primarily to scout:

- available tasks;
- episode lengths;
- possible source trajectory IDs.

A source is primary-compatible only after passing Section 6.

---

## Tier 1B — RoboVerse CALVIN Task Trajectory Mirror

Public source:

```text
RoboVerseOrg/roboverse_data
```

RoboVerse contains CALVIN task-organized trajectory files and environment-organized CALVIN trajectories.

Do not download the full RoboVerse repository.

Search/download only CALVIN trajectory paths.

Start with the six frozen tasks:

```text
lift_blue_block_slider
lift_red_block_table
place_in_slider
push_pink_block_right
turn_off_lightbulb
turn_on_lightbulb
```

Known task-organized CALVIN paths should be discovered programmatically from the Hub tree rather than guessed.

Where available, prefer Franka trajectory files such as:

```text
.../<task_name>_a/v2/franka_v2.pkl.gz
```

Also inspect:

```text
trajs/calvin/calvin_traj_ann/
trajs/calvin/env_D_out/
trajs/calvin/env_D_val_out/
```

Download only candidate files for the six tasks.

Use `huggingface_hub` APIs to enumerate repository files before downloading.

Example Python logic:

```python
from huggingface_hub import list_repo_files, hf_hub_download

repo = "RoboVerseOrg/roboverse_data"
files = list_repo_files(repo, repo_type="dataset")

wanted = [
    f for f in files
    if f.startswith("trajs/calvin/")
    and any(task in f for task in SIX_TASKS)
]
```

Do not assume RoboVerse trajectories are long enough.

Audit every downloaded object for:

- number of trajectories;
- number of steps per trajectory;
- action representation;
- robot embodiment;
- time base;
- task label;
- continuity.

RoboVerse-migrated CALVIN trajectories may still inherit original language segmentation. If they remain shorter than 160 compatible frames, they are not sufficient for the primary long-horizon benchmark.

---

## Tier 2 — Selective Original-Format CALVIN ABCD Subsets

Public source:

```text
VyoJ/calvin-ABCD-D-subsets
```

This mirror contains the original-style CALVIN structure:

```text
scene_info.npy
lang_annotations/auto_lang_ann.npy
ep_lens.npy
ep_start_end_ids.npy
episode_XXXXXXX.npz
```

It is divided into 24 training subsets plus validation.

Most training subsets are approximately 26–28 GB; the last is much smaller.

Do NOT download all subsets simultaneously.

### 2.1 Subset Order

Start with:

```text
subset_training_023
```

because it is the smallest training shard.

Then process:

```text
subset_training_000
subset_training_001
...
subset_training_022
```

one at a time until the long-data adequacy gate is satisfied.

Do not continue downloading after enough prospective long segments have been found.

### 2.2 Download One Subset

Preferred:

```bash
hf download VyoJ/calvin-ABCD-D-subsets \
  training/subset_training_023.zip \
  --repo-type dataset \
  --local-dir <STAGING_ROOT>
```

Fallback:

```bash
huggingface-cli download VyoJ/calvin-ABCD-D-subsets \
  training/subset_training_023.zip \
  --repo-type dataset \
  --local-dir <STAGING_ROOT>
```

or:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="VyoJ/calvin-ABCD-D-subsets",
    filename="training/subset_training_023.zip",
    repo_type="dataset",
    local_dir=STAGING_ROOT,
)
```

### 2.3 Extract and Audit

After extraction, inspect `auto_lang_ann.npy`.

Find every annotation matching one of the six canonical tasks.

For every match record:

```text
source_repo
subset
environment if recoverable
canonical_task
raw_language
start_frame
end_frame
inclusive_frame_count
contiguous
contains_other_annotation_boundary
```

Primary direct eligibility requires:

```text
frame_count >= 160
```

because:

```text
10 non-overlapping windows × H16 = 160 frames
```

Do not extend an annotation backward/forward simply to make it long.

Do not concatenate separate demonstrations.

Do not merge repeated same-language annotations.

If a candidate is eligible, retain the exact required `episode_XXXXXXX.npz` files plus metadata.

Then remove unnecessary extracted frames and the ZIP before moving to the next shard.

### 2.4 A/B/C Environment Data

A/B/C trajectories may be considered because the frozen action representation is action-only, but they still require exact action/time compatibility.

Do not assume cross-environment compatibility.

Pass Section 6 first.

Report source environment in every analysis.

---

## Tier 3 — Official CALVIN D→D Download Only If Missing Locally

Official CALVIN provides the D→D split through its download script.

If and only if a complete official D→D dataset is not already local:

```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh D
```

Run the official checksum procedure after download.

Do not duplicate the existing D/D dataset.

Official D→D is not expected by itself to solve the long-language-segment limitation if it is already the source used in prior waves; this tier is mainly for completeness/provenance.

---

# 3. Explicitly Excluded Sources for Primary Evaluation

## L-CALVIN / Long-VLA

Audit availability only.

Do not rely on L-CALVIN unless actual downloadable trajectory data and exact action/time compatibility are publicly available.

If only project-page/paper descriptions or short 64-frame re-annotations are available, mark:

```text
not_usable_for_H8_primary
```

Do not reconstruct unpublished benchmark data.

## Generic robot datasets such as DROID / BridgeData / LIBERO

Do not use them for frozen F1/F2 primary evaluation.

The embodiment, action parameterization, control frequency, and task semantics differ.

Using them would require representation/dynamics adaptation, which is forbidden in this prospective frozen-model experiment.

---

# 4. Open-Data Acquisition Audit

Before evaluating F1/F2, produce:

`open_data_source_audit.json`

For every source:

```text
source_name
repo_id
license
downloaded_bytes
temporary_bytes
task_coverage
trajectory_count
length_distribution
action_dim
action_semantics
control_frequency
robot
coordinate_frame
language_annotation_type
candidate_count_ge_160
compatibility_status
rejection_reason
```

Status must be exactly one of:

```text
PRIMARY_COMPATIBLE
SCOUTING_ONLY
REJECTED
```

Do not evaluate F1/F2 on SCOUTING_ONLY or REJECTED data.

---

# 5. Canonical Six-Task Mapping

Use the frozen six tasks:

```text
lift_blue_block_slider
lift_red_block_table
place_in_slider
push_pink_block_right
turn_off_lightbulb
turn_on_lightbulb
```

Recover all natural-language paraphrases from the frozen CALVIN task mapping.

Map open-data task labels to canonical tasks using metadata/task IDs when possible.

Do not use a semantic embedding classifier to invent mappings when exact task metadata exists.

For ambiguous labels:

```text
reject_ambiguous_task_mapping
```

Save:

`canonical_task_mapping.json`

before F1/F2 evaluation.

---

# 6. Hard Frozen-Model Data Compatibility Gate

A candidate trajectory can enter the primary benchmark only if ALL criteria pass.

## 6.1 Robot/action dimensionality

Required:

```text
Franka/Panda-compatible CALVIN action stream
7 action dimensions
```

## 6.2 Relative action convention

The action must match the frozen representation's exact CALVIN `rel_actions` convention.

Verify numerically and from source:

```text
dims 0:3 = relative TCP translation
dims 3:6 = relative TCP Euler rotation
dim 6 = gripper
```

Verify exact scaling/clipping used by the original frozen representation.

Do not silently convert absolute actions to relative actions unless the transformation is exact, source state is sufficient, and the conversion is preregistered before model outputs are inspected.

Preferred primary data already contains original-compatible `rel_actions`.

## 6.3 Frame rate

Required primary time base:

```text
30 Hz original CALVIN timing
```

or an exactly proven source-equivalent stream.

A 10-Hz converted LeRobot source must not be upsampled to 30 Hz for primary evaluation.

Do not repeat/interpolate actions to manufacture missing frames.

## 6.4 Continuity

Required:

```text
contiguous source frames
no missing frame IDs
no reset inside segment
no concatenated demos
```

## 6.5 Annotation consistency

Required:

```text
one canonical task / language context
for the entire >=160-frame segment
```

Do not cross language/task boundaries.

## 6.6 Length

Required:

```text
>=160 contiguous compatible frames
```

Preferred:

```text
>=176 frames
```

to provide additional valid rollout starts after conditioning windows.

## 6.7 No model-dependent filtering

Candidate inclusion must depend only on:

- source metadata;
- task mapping;
- length;
- continuity;
- action/time compatibility.

Do not use F1/F2 error to select trajectories.

Write the eligible set before inference.

---

# 7. Data Adequacy Gate

Target:

```text
6 tasks × 10 segments/task = 60 primary segments
```

Required:

```text
>=60 total PRIMARY_COMPATIBLE segments
>=10 per each of the six tasks
>=160 frames each
```

If exact per-task availability makes 10/task impossible, do NOT silently lower the criterion.

Instead produce:

`data_adequacy_failure.json`

and stop primary inference.

At that point list only the missing tasks/counts.

Manual CALVIN collection becomes a targeted fallback for only the missing cells.

Do not collect tasks already sufficiently covered by open data.

---

# 8. Prospective Evaluation Manifest

Before any F1/F2 metrics are computed, write:

`long_horizon_open_data_preregistration.json`

Include:

- exact source files;
- SHA256 hashes;
- exact 60+ selected segments;
- source/environment;
- canonical task;
- frame ranges;
- number of non-overlapping windows;
- frozen representation hash;
- frozen semantic predictor hash;
- F1 hash;
- F2 hash;
- F2 iteration count;
- horizons;
- metrics;
- bootstrap procedure;
- mechanism criteria;
- statement that no F1/F2 output on these segments has been accessed.

Only then may inference begin.

---

# 9. Source Stratification

If primary data comes from multiple sources/environments, do not only report pooled metrics.

Report:

```text
pooled
per-source
per-environment
per-task
```

The primary paired F1/F2 comparison can pool compatible trajectories, but bootstrap must preserve trajectory identity.

Also run a source-stratified bootstrap if more than one major source contributes.

A positive claim should not be driven solely by one source while reversing on another major source.

---

# 10. Frozen F1/F2 Evaluation

Use exact wave-15 frozen checkpoints.

Required:

```text
model parameter updates = 0
backward calls = 0
optimizer steps = 0
```

F1 and F2 must receive identical causal information.

F2:

```text
e0 = F1 prediction
e_final = frozen matched refinement(e0)
```

Use exact frozen iteration count/objective.

No future target actions.

---

# 11. Latent Construction

Primary:

```text
H = 16
stride = 16
```

No overlapping adjacent primary windows.

For each segment serialize:

```text
source
task
raw frame indices
window indices
z_sem
z_exec
full latent
language/context ID
valid rollout starts
```

Do not renormalize the representation on new data.

Use frozen normalization only.

---

# 12. Horizons

Evaluate exactly:

```text
H1
H2
H4
H8
```

where these denote latent rollout steps.

At 30 Hz and 16 frames per latent step, report exact physical duration.

No padding.

No ground-truth latent teacher forcing after rollout begins.

No cross-boundary continuation.

---

# 13. Primary Metrics

At each horizon report:

## Execution latent

- MSE
- normalized MSE
- cosine similarity

## Full latent

- full MSE
- semantic MSE
- execution MSE

## Decoded action

- continuous-action MSE
- gripper accuracy
- per-action-dimension errors

## Semantic retention

- latent→text retrieval
- correct task assignment
- semantic cosine

## Off-manifold

- execution nearest-training-latent distance
- execution kNN radius
- full-latent kNN radius
- ratio to ground-truth radius
- fraction beyond frozen train-derived threshold

---

# 14. Primary Statistical Endpoint

For every trajectory compute its own normalized rollout-error AUC over:

```text
1,2,4,8
```

Define:

```text
Delta_AUC = AUC_F2 - AUC_F1
```

Lower is better.

Use:

```text
paired whole-trajectory bootstrap
10,000 replicates
95% CI
```

Primary long-horizon claim requires:

```text
upper_95_CI(Delta_AUC) < 0
```

Do not bootstrap latent windows as independent samples.

Also report task-stratified and source-stratified bootstrap summaries.

---

# 15. Horizon-Specific Gate

Require:

```text
F2 H4 execution MSE < F1 H4 execution MSE
F2 H8 execution MSE < F1 H8 execution MSE
F2 H8 decoded continuous MSE < F1 H8 decoded continuous MSE
F2 H8 execution kNN radius < F1 H8 execution kNN radius
```

If F2 only wins at H1/H2:

```text
C3c-long = REJECTED
```

---

# 16. Refinement Intermediate-State Logging

For F2 save:

```text
e^(0) = F1
e^(1)
...
e^(K) = final refinement
```

At each iteration measure:

- GT execution-latent error;
- decoded-action error;
- nearest-manifold distance;
- kNN radius.

Do not feed GT information into refinement.

GT is evaluation only.

---

# 17. Correction Alignment

Evaluate:

```text
delta_refine = e_F2 - e_F1
delta_target = e_GT - e_F1
cosine(delta_refine, delta_target)
```

Report:

- mean;
- median;
- fraction >0;
- by horizon;
- by task;
- by source.

Target correction is never an input.

---

# 18. Empirical Executable-Manifold Audit

Build the empirical manifold reference using only the previously frozen dynamics-training latents.

Do not build it from the new long-horizon evaluation set.

For every F1 prediction:

1. retrieve fixed-k nearest training `z_exec`;
2. fit local PCA;
3. use a training-only frozen variance rule for tangent dimension;
4. compute normal distance;
5. decompose F2 correction into tangent/normal components.

Report:

- F1 normal distance;
- F2 normal distance;
- normal-distance reduction;
- tangent motion;
- correction component toward local manifold;
- association between normal-distance reduction and decoded-action improvement.

Call this:

```text
empirical executable latent manifold
```

not a physical manifold.

---

# 19. Mechanism Gate

Strong manifold-restoration claim requires all:

1. F2 lower execution kNN radius than F1;
2. F2 lower decoded-action MSE;
3. mean correction-target cosine >0;
4. mean empirical normal distance decreases;
5. normal-distance reduction is positively associated with decoded-action improvement.

If rollout improves but mechanism gate fails:

```text
C3c-long may be SUPPORTED
C3d manifold restoration = NOT_SUPPORTED
```

---

# 20. Frozen DEL Negative Baseline

Retain one historical frozen DEL baseline only if exact input/time compatibility permits evaluation on the selected source.

Do not retrain.

Do not include DEL in the F1/F2 primary bootstrap.

If source incompatibility prevents safe DEL evaluation:

```text
historical_DEL_negative_baseline = historical_context_only
```

Do not force it.

No DEL rescue.

---

# 21. Open-Data Failure Fallback

If open data cannot satisfy the 60-segment adequacy gate:

Do not change model claims.

Do not evaluate on an underpowered convenience set as the primary experiment.

Generate a precise targeted acquisition plan:

```text
task
open-data valid count
missing count to reach 10
minimum frames
required CALVIN-compatible action format
```

Then manual/VR collection should collect only the missing trajectories.

This is the only condition under which new CALVIN collection is authorized.

---

# 22. Final Claim Decision

Write:

`wave16_claim_decision.json`

Statuses:

```text
C1 = SUPPORTED
C2 = SUPPORTED
C3a_full_DEL = REJECTED
C3b_exec_DEL = REJECTED
C3c_local_refinement = SUPPORTED
C3c_long_refinement = SUPPORTED or REJECTED or NOT_TESTED_INSUFFICIENT_DATA
C3d_empirical_manifold_restoration = SUPPORTED or NOT_SUPPORTED or NOT_TESTED
```

---

# 23. Final Paper Story Rules

## If C3c-long + C3d pass

Use:

> Language grounds meaningful and executable action coordinates; free latent prediction proposes motion, while iterative refinement corrects accumulated drift toward empirically executable regions.

Short:

> Language tells actions where to live; refinement keeps their evolution executable.

Chinese:

> 语言为动作 latent 提供有意义且可执行的坐标锚点；自由预测负责推进，而迭代 refinement 持续修正累积漂移，使轨迹保持在经验上的可执行区域附近。

## If C3c-long passes but C3d fails

Use:

> Language-grounded action coordinates support predictive dynamics, and matched iterative refinement improves long-horizon rollout stability over an unconstrained latent predictor.

Do not claim manifold restoration.

## If C3c-long fails

Use:

> Language-grounded action coordinates are semantically addressable, executable, and locally predictable; refinement improves short-horizon prediction, while stable long-horizon latent dynamics remains unresolved.

---

# 24. Required Tests

Add tests for:

- local-data inventory;
- Hugging Face source enumeration;
- download hash verification;
- disk free-space guard;
- staged subset cleanup;
- exact task mapping;
- action dimensionality;
- exact rel-action convention;
- time-base compatibility;
- no 10-Hz→30-Hz interpolation;
- contiguous frames;
- >=160-frame segment eligibility;
- no annotation-boundary crossing;
- no model-dependent data selection;
- preregistration before inference;
- frozen F1/F2 hashes;
- no optimizer/backward;
- H=16 stride=16;
- H8 support;
- no padding;
- no future target action;
- 10,000 whole-trajectory bootstrap;
- training-only manifold reference;
- all metrics finite.

Do not overwrite prior-wave artifacts.

---

# 25. Required Deliverables

Produce:

- `sixteenth_wave_results.md`
- `sixteenth_wave_next_experiment.md`
- `disk_budget.json`
- `local_calvin_inventory.json`
- `open_data_source_audit.json`
- `canonical_task_mapping.json`
- per-source download manifest
- SHA256 manifest
- staged-download cleanup log
- long-trajectory availability audit
- primary compatibility audit
- data-adequacy gate JSON
- targeted missing-data acquisition plan if needed
- prospective evaluation preregistration
- frozen checkpoint manifest
- long latent serialization manifest
- H1/H2/H4/H8 sample-count table
- F1/F2 latent metrics
- decoded-action metrics
- semantic-retention metrics
- off-manifold metrics
- per-source/per-task metrics
- trajectory-level paired bootstrap
- source-stratified bootstrap
- refinement iteration diagnostics
- correction-vector alignment
- local tangent/normal audit
- mechanism decision JSON
- DEL frozen negative-baseline note
- final claim decision JSON
- exact executed commands
- environment/provenance report
- files-changed report
- complete tests
- updated `RESEARCH_LOG.md`
- updated `NEXT_EXPERIMENT.md`

Final report must explicitly answer:

1. What compatible CALVIN data already existed locally?
2. Which public sources were audited/downloaded?
3. How many bytes did each source require?
4. Which sources were rejected for time-base/action incompatibility?
5. Did the 10-Hz LeRobot conversion remain scouting-only?
6. How many direct >=160-frame annotation-consistent candidates were found in each source?
7. How many valid segments were found for each of the six tasks?
8. Did open data alone reach 10/task and 60 total?
9. If not, exactly which tasks/segments still require new collection?
10. Were F1/F2 frozen before all prospective long-data metrics?
11. How many rollout starts supported H1/H2/H4/H8?
12. Did F2 beat F1 in paired trajectory-level AUC with the 95% CI below zero?
13. Did F2 beat F1 at H4?
14. Did F2 beat F1 at H8?
15. Did F2 reduce H8 decoded-action error?
16. Did F2 reduce H8 execution off-manifold drift?
17. Was the refinement correction aligned with the GT correction?
18. Did empirical normal-to-manifold distance decrease?
19. Is C3c-long supported?
20. Is C3d supported?
21. Does DEL remain a frozen negative baseline only?
22. What final paper story is scientifically defensible?
23. Is any manual/VR CALVIN collection still necessary?

Do not write the desired conclusion before the open-data audit and prospective evaluation are complete.

Scientific correctness, data compatibility, prospective model freezing, and disk safety are the priority.
