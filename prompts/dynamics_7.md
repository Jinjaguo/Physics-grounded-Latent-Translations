# PGLT Nineteenth-Wave Codex Prompt
# Independent LIBERO-Long / LIBERO-10 Replication with Prospectively Branchable π0.5 Demonstrations

## 0. Mission

Wave 19 is a **genuinely independent embodied-domain replication** of the paper's current positive story.

The current paper story is:

> **Language grounds meaningful and executable action coordinates. A free predictor advances the latent trajectory, while iterative refinement suppresses accumulated drift and keeps long-horizon evolution near empirically executable regions.**

Wave 19 must test whether the same scientific phenomenon can be reproduced in a second embodied benchmark using:

```text
new simulator/domain: LIBERO
new source demonstrations: generated prospectively by π0.5
new representation: trained from scratch on LIBERO
new F1: trained from scratch on LIBERO
new F2: trained from scratch on LIBERO
same scientific protocol and claim logic
exactly branchable simulator states collected prospectively
```

π0.5 is **only a source-trajectory/data-collection policy**.

It plays the role that human teleoperation or robot demonstrations would play in a real-world data collection pipeline.

π0.5 is not the proposed method.

π0.5 is not F1.

π0.5 is not F2.

Do not use π0.5 internal hidden states as the latent representation.

The learned action coordinate remains:

```text
action chunk -> continuous action latent
```

The primary Wave-19 question is:

> **Does the language-grounded continuous action-coordinate + matched iterative-refinement phenomenon replicate independently on LIBERO, and does F2 improve exact-state closed-loop continuation when evaluated counterfactually from identical branch states?**

---

# 1. Naming Audit Before Anything Else

Do not assume that the installed benchmark calls the target suite "LIBERO-Long".

The official LIBERO repository historically exposes suites including:

```text
LIBERO-Spatial
LIBERO-Object
LIBERO-Goal
LIBERO-100
LIBERO-90
LIBERO-10
```

The project may internally refer to LIBERO-10 as "LIBERO-Long" because these are longer-horizon multi-step tasks.

Before downloading data or collecting trajectories:

1. inspect the exact installed LIBERO version;
2. enumerate available benchmark suites;
3. enumerate the exact 10 target tasks intended for this wave;
4. record task names, BDDL files, language instructions, success predicates, and controller configuration;
5. save the resolved terminology.

Write:

`wave19_libero_suite_audit.md`

The report must explicitly state:

```text
requested_internal_name = LIBERO-Long
resolved_official_suite_name = ...
number_of_tasks = ...
exact_task_list = [...]
```

If the requested target is not exactly LIBERO-10, use the actual installed suite selected by the repository configuration and document why.

Do not silently substitute another task suite.

---

# 2. Frozen Cross-Domain Scientific Protocol

Wave 19 is an **independent replication**, not zero-shot transfer.

Therefore:

```text
CALVIN representation is NOT reused
CALVIN F1 is NOT reused
CALVIN F2 is NOT reused
CALVIN normalization stats are NOT reused
```

Train a new LIBERO instance using the same architecture/protocol family.

The following scientific design choices should be kept fixed unless an interface incompatibility makes them impossible:

```text
factorized latent:
z = [z_sem, z_exec]

latent dimension:
32 total

semantic dimension:
16

execution dimension:
16

action-chunk horizon:
prefer H_action = 16 environment control steps

semantic alignment:
frozen text encoder + contrastive alignment

decoder:
full latent -> continuous action chunk

F1:
free execution-latent predictor

F2:
exact F1 proposal initialization
exact matched iterative refinement
four refinement iterations

DEL:
DO NOT RUN
DO NOT RETRAIN
DO NOT RESCUE
```

Any necessary action-interface adaptation must be resolved **before training** and written into a frozen compatibility manifest.

---

# 3. Official Source Repositories and Environment Setup

Use official/primary sources wherever possible.

## 3.1 LIBERO

Preferred repository:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

Record:

```bash
cd LIBERO
git rev-parse HEAD
git status --short
```

Save:

`provenance/libero_git_commit.txt`

Install according to the repository's current official instructions.

Do not hard-code old dependency versions before reading the checked-out repository.

## 3.2 OpenPI

Preferred repository:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
```

Record:

```bash
cd openpi
git rev-parse HEAD
git status --short
```

Save:

`provenance/openpi_git_commit.txt`

Inspect:

```text
examples/libero/README.md
```

and the current configuration defining:

```text
pi05_libero
```

The official OpenPI LIBERO example documents a checkpoint at:

```text
gs://openpi-assets/checkpoints/pi05_libero/
```

Do not blindly assume an old command-line interface.

Read the current README and use the exact commands supported by the checked-out version.

Save the exact download/evaluation commands in:

`provenance/pi05_download_and_eval_commands.sh`

## 3.3 π0.5 checkpoint provenance

Record:

- checkpoint URI;
- local checkpoint path;
- checkpoint hash if files are local;
- OpenPI commit;
- config name;
- normalization assets;
- LIBERO action postprocessing;
- observation preprocessing;
- action chunk size;
- action horizon;
- inference server/client configuration;
- random seeds.

Write:

`wave19_pi05_source_policy_manifest.json`

Do not fine-tune π0.5 during Wave 19 unless the exact official LIBERO checkpoint is unavailable.

If fine-tuning becomes necessary, STOP and report.

Wave 19 is intended to use a fixed source-data generator.

---

# 4. Storage Layout

Create one root:

```text
data/wave19_libero_branchable/
```

Recommended layout:

```text
data/wave19_libero_branchable/
├── provenance/
│   ├── libero_git_commit.txt
│   ├── openpi_git_commit.txt
│   ├── environment_freeze.txt
│   ├── pi05_source_policy_manifest.json
│   └── collection_config.json
│
├── raw_collection/
│   ├── task_00/
│   │   ├── successes/
│   │   │   ├── episode_000000/
│   │   │   └── ...
│   │   └── failures/
│   └── ...
│
├── certified/
│   ├── task_00/
│   │   ├── episode_000000/
│   │   │   ├── episode_metadata.json
│   │   │   ├── instruction.txt
│   │   │   ├── actions.npy
│   │   │   ├── robot_states.npz
│   │   │   ├── object_states.npz
│   │   │   ├── observations/
│   │   │   ├── branches/
│   │   │   │   ├── branch_025/
│   │   │   │   ├── branch_050/
│   │   │   │   └── branch_075/
│   │   │   └── source_continuation/
│   │   └── ...
│   └── ...
│
├── derived/
│   ├── representation/
│   ├── dynamics/
│   └── manifests/
│
└── audits/
```

Never modify `raw_collection/` after an episode is finalized.

Derived latents/checkpoints must live outside raw source folders.

---

# 5. Disk Guard

Before collection:

```text
minimum free disk = 300 GB
preferred free disk = 500 GB+
```

Before each major phase, record:

```bash
df -B1 .
```

Write:

`audits/disk_usage_log.jsonl`

Do not save redundant RGB/depth streams if not needed for source-policy reproducibility, but do not omit any simulator state required for exact branch restoration.

Exact state is more important than image archival.

---

# 6. π0.5 Source-Trajectory Collection

## 6.1 Role of π0.5

π0.5 should generate successful source demonstrations:

```text
observation + language
        ↓
      π0.5
        ↓
  environment action
        ↓
  LIBERO simulator
```

These successful trajectories act as the equivalent of human demonstrations.

They define source paths from which exact counterfactual branch states are later created.

## 6.2 Collect all outcomes

Do not discard failures during raw collection.

Store:

```text
successes/
failures/
```

Primary Wave-19 representation/dynamics source data should use successful trajectories unless otherwise preregistered.

Failure trajectories may be used for secondary analysis only.

## 6.3 Target scale

Primary collection target:

```text
10 tasks
30 successful independent episodes per task
= 300 successful source episodes
```

Absolute minimum:

```text
20 successful episodes per task
= 200 total
```

Preferred if compute permits:

```text
50 successful episodes per task
= 500 total
```

Do not compensate for a difficult task by heavily oversampling easy tasks in the primary benchmark.

Primary benchmark should remain task-balanced.

## 6.4 Independence

Each source episode must begin from a fresh LIBERO environment reset with independently sampled task-valid initialization.

Record:

```text
environment seed
task seed
policy seed if applicable
initial simulator state hash
language instruction
task identifier
```

Do not obtain multiple "independent episodes" by branching one trajectory.

---

# 7. What Must Be Saved at Every Environment Step

Wave 18 failed because public CALVIN files lacked exact dynamic/controller/contact state.

Wave 19 must solve this prospectively.

For every environment step save the strongest exact simulator representation available.

Because LIBERO is built on robosuite/MuJoCo-style simulation, inspect the exact installed API rather than assuming names.

At minimum capture:

## 7.1 Raw simulator state

Prefer native MuJoCo state serialization / full state copy.

Save all state required to reproduce dynamics, including as available:

```text
qpos
qvel
act
ctrl
mocap_pos
mocap_quat
userdata if used
time
warmstart/solver state if required for exact restore
```

If the installed MuJoCo API exposes an official complete state specification, use it.

Do not manually save only qpos/qvel if more state is required for deterministic restore.

## 7.2 Robot/controller state

Save:

```text
robot joint positions
robot joint velocities
gripper joints
gripper command
end-effector pose
controller goals/targets
controller internal state
action scaling parameters
previous action if used
action interpolation state if used
```

## 7.3 Object/scene state

Save:

```text
all movable object poses
all movable object linear velocities
all movable object angular velocities
articulated object joint qpos/qvel
drawer/cabinet/door states
site/body transforms if required
```

## 7.4 Task/logical state

Save:

```text
task id
BDDL path
language instruction
official success predicate result
task-relevant object identities
episode step
```

## 7.5 Observation and action

Save:

```text
raw policy observation
policy-processed observation if practical
π0.5 output before environment postprocessing
actual action passed to env.step
postprocessed/clipped action
reward if available
done
success
```

This distinction is mandatory.

Do not conflate network output with executed action.

---

# 8. Action Mutation Safety

Wave 18 exposed an important implementation hazard: some environment/control pipelines may mutate action arrays in-place.

Therefore:

```python
env.step(action.copy())
```

or an equivalent immutable-action boundary must be enforced throughout collection and replay.

Add a unit test:

```text
input action bytes/hash unchanged after env.step wrapper
```

If the underlying environment mutates the array, the wrapper must copy before passing it downstream.

Record this behavior in:

`audits/action_mutation_audit.md`

---

# 9. Branchable Snapshot Design

For every successful source episode create candidate branches at fixed fractions:

```text
25%
50%
75%
```

Primary branch selection is based on source episode progress only.

Do not move branch points based on F1/F2 output.

A branch is eligible only if sufficient future source steps remain for the requested evaluation horizon.

For H8 with H_action=16:

```text
required future control steps >= 128
```

If the LIBERO control frequency/action semantics make H=16 inappropriate, resolve this **before model training** using the timebase audit in Section 13.

Each branch directory must contain:

```text
branch_025/
├── branch_metadata.json
├── exact_sim_state.bin / .npz
├── controller_state.pkl / .npz
├── causal/
│   ├── past_actions.npy
│   ├── current_instruction.txt
│   ├── issue_step.json
│   └── causal_availability.json
└── reference_only/
    ├── future_actions.npy
    ├── future_robot_states.npz
    ├── future_object_states.npz
    └── source_terminal_success.json
```

Keep `causal/` and `reference_only/` physically separated.

Formal model input code must never read from `reference_only/`.

---

# 10. Snapshot Certification Gate

Every candidate branch must be certified before it can enter the benchmark.

This gate is mandatory and occurs before representation/F1/F2 evaluation.

For each branch:

1. restore the snapshot into Twin A;
2. independently restore the same snapshot into Twin B;
3. replay the same recorded π0.5 source continuation;
4. compare A/B at every step;
5. compare both restored trajectories to the source continuation;
6. evaluate the official task predicate.

Required outputs:

```text
continuous state discrepancy
controller state discrepancy
object state discrepancy
terminal predicate agreement
replay success agreement
nonfinite count
```

Determine deterministic tolerances from simulator repeated-restore tests **before model inference**.

Do not reuse Wave-18's 1e-9 threshold automatically if MuJoCo determinism has a different natural numerical floor.

Protocol:

```text
run >=100 repeated restore-and-replay trials
on a development-only branch set

measure numerical discrepancy distribution

freeze:
median tolerance
P95 tolerance
terminal predicate requirement
```

The tolerances must be based only on simulator determinism.

They must not depend on F1/F2 results.

Write:

`wave19_snapshot_certification_preregistration.json`

A certified branch requires:

```text
100% twin terminal-predicate agreement
100% restored-vs-source terminal-predicate agreement
all values finite
median discrepancy <= frozen median tolerance
P95 discrepancy <= frozen P95 tolerance
```

If exact-enough branch restoration cannot be achieved:

STOP.

Do not run closed-loop F1/F2 evaluation.

Write:

`wave19_reconstruction_gate_failure.md`

---

# 11. Dataset Freeze and Split

After enough certified source episodes exist, freeze a manifest before representation training.

Because this is an independent-domain replication, use episode-level splits.

Recommended:

```text
60% representation/dynamics train
20% development
20% final held-out test
```

Stratify by task.

No branch from the same source episode may appear in multiple splits.

No image/state/action chunk from the same source episode may cross splits.

Write:

`wave19_dataset_split_manifest.json`

Include SHA256 hashes for every source episode and branch metadata file.

The final held-out test must remain unread until all model/hyperparameter choices are frozen.

---

# 12. LIBERO Action-Interface Audit

Before training the representation, audit the exact action interface.

Record:

```text
action dimension
translation semantics
rotation representation
gripper semantics
action scaling
controller type
control frequency
action repeat
whether actions are relative/absolute
clipping
normalization
```

Write:

`wave19_libero_action_interface.md`

Do not assume compatibility with CALVIN merely because both may use 7-D controls.

The LIBERO representation is trained from scratch, so exact CALVIN compatibility is unnecessary.

However, the scientific architecture must still operate on a well-defined continuous action coordinate.

---

# 13. Timebase and Chunk-Horizon Audit

Determine actual LIBERO control frequency.

Let:

```text
f_control = verified environment control frequency
```

Prefer to preserve:

```text
H_action = 16 control steps
stride = 16
```

if this corresponds to a sensible manipulation timescale.

Report:

```text
latent_step_seconds = H_action / f_control
```

If 16 steps is clearly inappropriate, choose an alternative once on a development-only protocol before representation training.

Requirements:

- no overlap in primary dynamics windows;
- enough H8 duration to be genuinely long-horizon;
- no tuning based on final test performance.

Write:

`wave19_timebase_preregistration.json`

---

# 14. Train a New LIBERO Representation from Scratch

The LIBERO representation must be independent of CALVIN weights.

Architecture:

```text
action-only encoder
input = H_action consecutive executed LIBERO actions

z = [z_sem, z_exec]

dim(z_sem) = 16
dim(z_exec) = 16
```

Decoder:

```text
D(z_sem, z_exec) -> H_action continuous LIBERO actions
```

Text branch:

```text
frozen text encoder
language instruction -> text embedding
```

Loss:

```text
L_repr =
lambda_rec * L_action_reconstruction
+
lambda_sem * L_contrastive_semantic_alignment
```

Use the same gradient-isolation principle validated on CALVIN:

```text
semantic alignment should not update the shared motor trunk
if that is the frozen architecture used in the final CALVIN method
```

Use EMA with the same frozen decay used in the finalized method:

```text
EMA decay = 0.999
```

Do not retune EMA on final test.

---

# 15. LIBERO Representation Readiness Gate

Wave 19 needs its own independent R-Gate.

Do not copy CALVIN thresholds mechanically where scales differ.

Before final test, preregister LIBERO equivalents based on development-only statistics.

Must test:

## Semantic

```text
text-to-action R@1
action-to-text R@1
macro task retrieval
correct-language vs reconstruction-only
correct-language vs shuffled-language
```

Define:

```text
semantic_delta =
correct-language metric
-
max(reconstruction-only, shuffled-language)
```

Required:

```text
positive mean semantic delta in both directions
episode-clustered bootstrap lower 95% > 0
```

## Executability

Measure:

```text
continuous action MSE
per-dimension MSE
gripper accuracy / appropriate gripper metric
decoder saturation/clipping
```

Compare correct-language representation to reconstruction-only.

The language-conditioned representation must not obtain semantic gains by destroying motor fidelity.

## Multi-seed replication

Use at least:

```text
6 seeds
```

if computationally feasible.

Required:

```text
representation_ready_LIBERO = true
```

before any dynamics training.

If the gate fails:

STOP.

Do not train F1/F2.

Write:

`wave19_representation_gate_failure.md`

---

# 16. Train LIBERO F1 and F2 from Scratch

## F1 — Free execution-latent predictor

Train using non-overlapping latent windows from successful source trajectories.

Input should follow the same causal structure as the final CALVIN dynamics model.

At minimum:

```text
z_exec(k-1)
z_exec(k)
current semantic state/context
```

No future action.

No future robot state.

No future task-success information.

## F2 — Matched iterative refinement

F2 must start from the exact F1 proposal:

```text
e^(0) = F1(...)
```

Then apply exactly:

```text
4 refinement iterations
```

Use the same architecture/objective family as the final CALVIN F2.

Do not introduce a LIBERO-specific refinement mechanism after seeing results.

---

# 17. Training-Phase Baselines

Include:

```text
copy
constant velocity
F1 free predictor
F2 matched refinement
```

Optional:

```text
matched extra-compute generic baseline
```

Do not run DEL.

Historical DEL negative evidence remains a CALVIN finding.

Wave 19 is an independent positive replication, not another variational-mechanics search.

---

# 18. Offline LIBERO Dynamics Evaluation

Before closed-loop branch execution, reproduce the same offline evidence structure.

Evaluate recursively at:

```text
H1
H2
H4
H8
```

Primary metrics:

```text
execution latent MSE
full latent MSE
decoded continuous action MSE
gripper disagreement
execution kNN radius
full-latent kNN radius
local-PCA normal distance
normalized rollout error AUC
```

Mechanism metrics:

```text
correction-target cosine
fraction positive cosine
iteration 0->4 latent error
iteration 0->4 decoded error
iteration 0->4 kNN radius
iteration 0->4 normal distance
```

Bootstrap by source episode.

Do not treat windows as independent.

---

# 19. Offline Replication Gate

Before opening the closed-loop test:

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
final F2 rollout has lower empirical normal distance than F1
on average
```

If this gate fails:

```text
cross_domain_offline_replication = REJECTED
```

Still report results.

Do not retune F2.

---

# 20. Exact-State Closed-Loop Counterfactual Evaluation

This is the centerpiece of Wave 19.

For each certified held-out branch, restore the exact same simulator state separately for each method.

Counterfactual rollouts:

```text
B0 = source π0.5 continuation reference
B1 = F1
B2 = F2
B3 = norm-matched random refinement
B4 = shuffled learned-direction refinement
B5 = negative refinement
```

Every method begins from the exact same certified branch state.

---

# 21. Causal Warm Start

At a branch point k, provide F1/F2 with only causally observed action-coordinate history.

Construct:

```text
z_{k-1}
z_k
current language/context
```

from actions executed before the branch issue time.

No future π0.5 action is model input.

No future environment state is model input.

No future success predicate is model input.

Future source continuation exists only in `reference_only/`.

---

# 22. Recursive Model Execution

At each latent step:

1. use current model latent history;
2. F1 predicts next execution latent;
3. F2 initializes from the exact F1 proposal;
4. F2 applies four frozen refinement iterations;
5. combine with the frozen/shared semantic prediction;
6. decode to H_action LIBERO environment actions;
7. execute each action in the restored simulator;
8. record actual environment state and success predicate;
9. recursively continue from model-generated latent history.

Important:

The action-coordinate model does not observe current RGB/state during continuation.

Do not secretly feed simulator observation into F1/F2.

This is a test of **latent continuation capacity**, not a replacement VLA.

π0.5 is allowed to remain observation-conditioned only in the source/reference branch.

---

# 23. Mandatory Refinement Causal Controls

## B3 — Norm-matched random

At each refinement step:

```text
||delta_random|| = ||delta_F2||
```

Use a random direction in normalized execution-latent space.

Freeze random seeds before final test.

## B4 — Shuffled learned direction

Use a learned correction direction from an unrelated development/training source state.

Norm-match to the target F2 update.

Never draw shuffled directions from the held-out branch itself.

## B5 — Negative refinement

Apply the negative learned correction direction.

Define exact recurrence before final test.

All controls use:

```text
same F1 initial proposal
same number of refinement iterations
same branch state
same action decoder
same horizon
```

---

# 24. Latent Proposal Perturbation Recovery

Perform a separate causal stress test.

At the F1 proposal:

```text
e_noisy = e_F1 + sigma * epsilon
```

Use frozen training execution-latent statistics.

Recommended normalized scales:

```text
sigma ∈ {0.05, 0.10, 0.20} × train execution std
```

Compare:

```text
F1 noisy
F2 from noisy proposal
random refinement from noisy proposal
negative refinement from noisy proposal
```

Primary question:

> Does learned refinement enlarge the basin over which a perturbed proposal can return to a successful/executable continuation?

---

# 25. Closed-Loop Horizons

Evaluate:

```text
H1
H2
H4
H8
```

in latent steps.

Also evaluate:

```text
until source horizon / task success / environment termination
```

as a separate endpoint.

Report exact seconds based on verified LIBERO control frequency.

---

# 26. Primary Closed-Loop Endpoints

## 26.1 Official task success

Use LIBERO's official task-success evaluator.

Primary outcome:

```text
success_F2 - success_F1
```

Report:

```text
overall
per task
per branch fraction
per horizon
```

## 26.2 Success preservation ratio

Because π0.5 source trajectories are successful, define:

```text
continuation_preservation(method)
=
fraction of certified successful source branches
that remain successful under method continuation
```

This is a useful branch-level embodied metric.

## 26.3 Time to failure / success

Where well-defined, report:

```text
time to official success
time until irreversible divergence if a valid predicate exists
```

Do not invent arbitrary failure predicates.

---

# 27. Physical Execution Diagnostics

Record during every rollout:

```text
robot q
robot dq
TCP pose
gripper state
object poses
articulated object states
contacts if exposed
```

Compare to source continuation using:

```text
joint-space deviation
TCP position deviation
TCP orientation deviation
object pose deviation
gripper disagreement
```

Treat these as diagnostics.

Do not assume source imitation distance is equivalent to task success.

---

# 28. Latent-to-Behavior Chain

For each branch compute:

```text
F1 -> F2 reduction in latent error
F1 -> F2 reduction in decoded action error
F1 -> F2 reduction in kNN radius
F1 -> F2 reduction in local-PCA normal distance
F1 -> F2 physical trajectory improvement
F1 -> F2 task-success improvement
```

Use clustered associations to test the chain:

```text
latent stabilization
    ↓
decoded command improvement
    ↓
physical continuation improvement
    ↓
task success
```

Use cautious terminology:

```text
mechanism-outcome association
```

Do not claim formal causal mediation without assumptions.

---

# 29. Primary Statistics

Highest-level independent unit:

```text
source episode
```

Multiple 25/50/75% branches from one episode are nested.

Use:

```text
10,000 bootstrap replicates
cluster = source episode
seed = 190819
```

For success:

- paired source-episode clustered bootstrap;
- absolute success-rate difference;
- 95% CI;
- paired binary test if appropriate.

For continuous metrics:

- paired source-episode clustered bootstrap;
- 95% CI.

Never bootstrap windows as independent samples.

---

# 30. Cross-Task Breadth Gate

A positive LIBERO embodied claim should not be driven by one task.

Require, where all 10 tasks have adequate samples:

```text
F2 - F1 success difference >= 0 on at least 8/10 tasks

and

positive on at least 6/10 tasks
```

Also require overall clustered success CI to exclude zero in the positive direction.

If some tasks have insufficient certified branches, report a reduced breadth gate prospectively before final inference.

Do not modify after seeing results.

---

# 31. Final Wave-19 Claim Gates

## G-A — Independent representation replication

```text
LIBERO_C1_language_addressability = SUPPORTED
LIBERO_C2_action_executability = SUPPORTED
```

only if the representation readiness gate passes.

## G-B — Independent offline dynamics replication

```text
LIBERO_C3c_long = SUPPORTED
```

only if the offline replication gate passes.

## G-C — Closed-loop embodied refinement

```text
LIBERO_C4_closed_loop_refinement = SUPPORTED
```

only if ALL are true:

```text
F2 success > F1
clustered 95% CI excludes 0

F2 improves H4 embodied/decoded error

F2 improves H8 embodied/decoded error

F2 has lower embodied execution kNN radius

snapshot certification passed

all model/data manifests frozen

no future leakage
```

## G-D — Learned direction specificity

```text
LIBERO_C5_learned_direction_value = SUPPORTED
```

only if:

```text
F2 > norm-matched random
F2 > shuffled learned direction
F2 > negative refinement
correction-target alignment positive
```

## G-E — Perturbation recovery

```text
LIBERO_C6_proposal_recovery = SUPPORTED
```

only if F2 improves recovery relative to F1-noisy and matched controls.

---

# 32. Cross-Domain Paper Decision

After Wave 19 write:

`wave19_cross_domain_claim_decision.json`

Possible strongest conclusion:

```text
CALVIN:
C1 supported
C2 supported
C3c-local supported
C3c-long supported
C3d supported

LIBERO:
independent C1 supported
independent C2 supported
independent C3c-long supported
closed-loop refinement supported
direction-specific causal controls supported
```

If so, the paper may use:

> **Across two independent embodied benchmarks, language-grounded continuous action coordinates exhibit semantic and executable structure, while iterative refinement consistently stabilizes their dynamics. On prospectively branchable LIBERO trajectories, this latent stabilization further translates into improved closed-loop continuation from identical physical states.**

Chinese:

> **在两个独立 embodied benchmark 上，语言监督都能够形成具有语义结构且可解码执行的连续动作坐标；迭代 refinement 则稳定其 latent dynamics。进一步地，在 prospectively branchable 的 LIBERO 轨迹上，这种 latent 稳定化能够从完全相同的物理 branch state 转化为更可靠的闭环 continuation。**

---

# 33. What Must Not Be Claimed

Do not claim:

```text
zero-shot CALVIN -> LIBERO transfer
```

because LIBERO representation/F1/F2 are retrained independently.

Do not claim:

```text
π0.5 hidden-state refinement
```

because π0.5 is only the data generator.

Do not claim:

```text
F1/F2 is a full VLA policy
```

because continuation does not observe the environment after branching.

Do not claim:

```text
physical manifold
```

for kNN/local-PCA metrics.

Use:

```text
empirical executable latent region/manifold
```

Do not claim:

```text
every refinement iteration is a projection
```

unless the new data unexpectedly supports a strict monotonic theorem-level result, which is not required.

---

# 34. Failure Taxonomy

For closed-loop failures classify, using preregistered observable categories:

```text
wrong reach direction
gripper timing error
failed grasp
object drop
transport drift
collision
overshoot
under-travel
task-sequence mismatch
articulation failure
off-manifold excursion
action saturation
simulation/controller failure
other
```

Do not change categories based on which method wins.

Report F1->F2 repair transitions.

---

# 35. Publication Figures

Prepare raw CSV/JSON for:

## Figure 1
Two-domain overview:

```text
CALVIN
LIBERO
```

representation + dynamics + closed-loop evidence.

## Figure 2
LIBERO semantic/executable representation:

```text
bidirectional retrieval
action reconstruction
multi-seed R-Gate
```

## Figure 3
LIBERO offline H1/H2/H4/H8:

```text
F1 vs F2 latent error
decoded error
kNN radius
```

## Figure 4
Exact-state closed-loop success:

```text
π0.5 source reference
F1
F2
random
shuffled
negative
```

## Figure 5
Branch depth:

```text
25%
50%
75%
```

success and error.

## Figure 6
Perturbation recovery:

```text
sigma vs success/error
```

## Figure 7
Mechanism:

```text
iteration 0->4
correction cosine
kNN reduction
decoded-action improvement
closed-loop success
```

---

# 36. Required Unit / Integration Tests

At minimum test:

```text
LIBERO version/commit recorded
OpenPI version/commit recorded
π0.5 checkpoint manifest recorded

action array is not mutated across wrappers
executed action equals saved executed action

raw simulator state can be restored
controller state can be restored
twin replay deterministic within frozen tolerances
restored replay matches source continuation

branch source episode IDs unique
no branch crosses episode reset
25/50/75 selection deterministic
future-support eligibility correct

train/dev/test split is episode-disjoint
source episode hashes immutable

action chunking correct
primary windows non-overlap
timebase correct

representation initialized independently from CALVIN
LIBERO F1 initialized independently from CALVIN
LIBERO F2 initialized independently from CALVIN

F2 starts from exact F1 proposal
exactly four refinement iterations
random correction norm matched
shuffled direction comes from allowed split
negative correction sign correct

no future action leakage
no future state leakage
no reference_only file read by predictor

source-episode clustered bootstrap correct
official LIBERO success predicate used

all outputs finite
```

Target:

```text
all tests pass
```

---

# 37. Required Manifests

Create before corresponding final inference:

```text
wave19_libero_suite_audit.md
wave19_pi05_source_policy_manifest.json
wave19_collection_preregistration.json
wave19_snapshot_certification_preregistration.json
wave19_dataset_split_manifest.json
wave19_libero_action_interface.md
wave19_timebase_preregistration.json
wave19_representation_preregistration.json
wave19_representation_gate.json
wave19_dynamics_preregistration.json
wave19_closed_loop_preregistration.json
wave19_frozen_model_manifest.json
wave19_cross_domain_claim_decision.json
```

---

# 38. Required Final Deliverables

Produce:

```text
nineteenth_wave_results.md
nineteenth_wave_next_experiment.md

wave19_libero_suite_audit.md
wave19_data_collection_report.md
wave19_pi05_source_policy_manifest.json

wave19_snapshot_certification_report.md
wave19_snapshot_certification_results.json

wave19_dataset_manifest.json
wave19_dataset_split_manifest.json

wave19_representation_results.md
wave19_representation_gate.json

wave19_dynamics_results.md
wave19_closed_loop_results.md
wave19_intervention_results.md
wave19_perturbation_recovery.md

wave19_failure_taxonomy.md
wave19_statistical_report.md
wave19_cross_domain_claim_decision.json

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

# 39. Exact Command Logging

Every external command used for:

```text
clone
install
checkpoint download
dataset collection
representation training
F1 training
F2 training
offline evaluation
snapshot certification
closed-loop evaluation
statistics
tests
```

must be appended to:

```text
exact_commands.sh
```

with timestamps or phase comments.

Do not rely on shell history.

---

# 40. Download Failure Handling

If a download fails:

1. keep partial/log output;
2. retry only with the same official source;
3. do not silently switch to unofficial mirrors;
4. record checksum/size after completion.

For very large artifacts:

- verify free disk first;
- download only required checkpoints/assets;
- do not download unrelated LIBERO datasets if π0.5 is generating source data online.

If the official π0.5 LIBERO checkpoint cannot be retrieved:

STOP.

Do not substitute a different VLA without a new preregistration.

---

# 41. Stop Conditions

Stop and report immediately if any of the following occurs:

```text
target LIBERO suite cannot be unambiguously resolved

π0.5 official LIBERO checkpoint unavailable

π0.5 cannot achieve enough successful source trajectories

exact simulator/controller state cannot be restored reliably

snapshot certification gate fails

fewer than 200 successful certified source episodes can be collected

representation R-Gate fails

train/dev/test episode leakage detected

CALVIN weights accidentally loaded into LIBERO representation/F1/F2

future action/state/reference leakage detected

model/checkpoint hashes change after freeze

closed-loop evaluation requires adding observation feedback to F1/F2

official LIBERO task predicate cannot be reproduced
```

Do not redesign the method after a stop condition.

Write a failure report and preserve all artifacts.

---

# 42. Final Report Questions

The final Wave-19 report must answer:

1. What exact official LIBERO suite corresponds to the requested "LIBERO-Long" target?
2. What exact 10 tasks were used?
3. What OpenPI commit and π0.5 checkpoint generated the source trajectories?
4. How many raw successful and failed π0.5 episodes were collected per task?
5. How many successful source episodes passed snapshot certification?
6. Are restored twins deterministic?
7. Do restored source continuations reproduce the original source task success?
8. What exact LIBERO action interface and control frequency were used?
9. What H_action and latent-step duration were frozen?
10. Was the LIBERO representation trained completely independently from CALVIN?
11. Does LIBERO language supervision improve bidirectional semantic retrieval?
12. Does the same representation preserve action reconstruction?
13. Did the independent LIBERO representation R-Gate pass?
14. Did F2 beat F1 offline at H1/H2/H4/H8?
15. What is the source-episode clustered AUC difference and CI?
16. Does F2 lower H8 decoded action error?
17. Does F2 lower H8 execution kNN radius?
18. Is correction-target cosine positive?
19. Does F2 improve exact-state closed-loop continuation success relative to F1?
20. What is the paired clustered success-rate difference and CI?
21. Is F2 benefit broad across tasks?
22. Does F2 beat norm-matched random refinement?
23. Does F2 beat shuffled learned directions?
24. Does negative refinement hurt relative to F2?
25. Does F2 recover from latent proposal perturbations?
26. Does lower off-manifold drift associate with better embodied outcomes?
27. Which F1 failure modes are repaired by F2?
28. Which remain?
29. Which CALVIN claims replicate on LIBERO?
30. What exact cross-domain paper story is now defensible?
31. Is any additional experiment still necessary before paper submission?

---

# 43. Strategic Meaning

Wave 19 should transform the evidence structure from:

```text
CALVIN:
semantic
+
executable
+
offline local dynamics
+
offline long-horizon refinement
+
empirical manifold stabilization
```

into:

```text
CALVIN:
discovery + deep diagnosis

LIBERO:
independent representation replication
+
independent dynamics replication
+
prospective exact-state counterfactual closed-loop validation
```

π0.5 is deliberately used as a demonstration generator because this mirrors the role of real-world data collection:

```text
expert / human / VLA
        ↓
successful demonstrated behavior
        ↓
action-coordinate learning
        ↓
latent dynamics
        ↓
counterfactual branch evaluation
```

The key objective is not to show that π0.5 works.

The key objective is to show that the **same Latent Engineering phenomenon independently emerges in a new embodied domain and survives exact-state behavioral testing**.

Run the protocol prospectively.

Do not chase a desired positive result.
