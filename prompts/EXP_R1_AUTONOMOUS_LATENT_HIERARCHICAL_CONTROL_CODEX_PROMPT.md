# EXP_R1+ Autonomous Codex Research Program
## Hierarchical Latent Path Planning for Actions as Coordinates
## F1 Local Dynamics + F2 Latent MPC + F3 Task Executive and Waypoint Memory

# 0. Role and naming reset

You are Codex operating inside the existing **Actions as Coordinates** research repository.

This is a **new research program**.

Do not continue the old Wave naming scheme.

Do not create:

```text
Wave79
Wave80
waveXX
```

for this research direction.

The new experiment series begins at:

```text
EXP_R1
```

and may autonomously continue through at most:

```text
EXP_R80
```

Stop when either:

```text
SUCCESS
```

is reached, or after completing:

```text
EXP_R80
```

Never start `EXP_R81`.

You are authorized to iterate autonomously without asking for confirmation, provided that every experiment obeys the scientific, held-out, reporting, and audit rules below.

---

# 1. Read the revised direction first

Before changing code, read:

```text
ACTIONS_AS_COORDINATES_POST_WAVE78_RESEARCH_DIRECTION.md
```

Then read all relevant historical evidence, including the actual repository artifacts for:

```text
representation readiness / frozen representation
F1 training and evaluation
F2 refinement
continuous-play long-horizon evaluation
Wave21–Wave27 language and transition experiments
Wave28–Wave78 summary and failure taxonomy
project paper/story files
dataset manifests
tests
```

Do not rely on filenames guessed from this prompt.

Search the repository and inspect the actual source and artifacts.

Repository source and machine-readable artifacts are authoritative.

Do not guess tensor names, checkpoint paths, task IDs, state fields, action fields, JSON keys, or interfaces.

---

# 2. Historical conclusions that must be preserved

## 2.1 Representation

The accepted action representation should remain frozen initially.

Preserve its exact accepted configuration and hashes.

Do not silently change:

```text
latent dimensionality
semantic/execution factorization
action-window length
encoder
decoder
text tower
text projection
normalization
EMA checkpoint identity
```

Historical strict Gate-A failures must remain preserved in provenance even if later readiness evidence authorized dynamics.

Do not retrain the representation in EXP_R1.

## 2.2 F1

Existing F1 should initially be interpreted only as:

> **a learned local latent dynamics predictor / prior.**

It predicts where the current action trajectory naturally evolves next from causal information.

Do not upgrade the historical F1 claim into Action-A-to-Action-B planning.

## 2.3 Existing F2

Existing F2 should initially be interpreted as:

> **iterative local refinement that improves or stabilizes F1 proposals.**

It is not yet MPC.

It does not prove arbitrary action-to-action transition.

Retain the exact old F2 as a frozen baseline.

## 2.4 Language causality

Preserve the established finding that changing next-goal language can causally redirect predicted future latent dynamics, including execution coordinates.

Do not interpret this as evidence that pointwise steering is sufficient.

## 2.5 Wave28–Wave78

Preserve the broad negative result:

> **pointwise / residual / force-field latent steering did not reliably combine target redirect, continuity, and endpoint identity.**

Do not continue adapter stacking as the default research direction.

The new program is about **multi-step path planning**.

---

# 3. New central hypothesis

The new scientific hypothesis is:

> **Atomic action regions in the frozen language-addressable action latent can be connected by feasible multi-step latent trajectories when local dynamics, continuous trajectory optimization, and high-level subgoal switching are separated.**

The target hierarchy is:

```text
F1 — local latent dynamics prior

F2 — latent trajectory optimizer / MPC-like receding-horizon controller

F3 — task navigator + subgoal-completion manager + waypoint execution memory
```

The frozen decoder maps selected latent coordinates back to robot actions.

---

# 4. Module responsibilities

## F1 — Local Dynamics

F1 answers:

```text
If the current atomic behavior continues naturally,
where should the next latent approximately lie?
```

F1 must not initially solve:

```text
long-horizon task decomposition
subgoal completion
lift -> place switching
return
global path planning
```

Use the existing frozen F1 in EXP_R1 unless the actual repository interface makes this impossible.

## F2 — Latent Path Planner

F2 answers:

```text
Given:
- the current executable latent;
- a target atomic-action region;
- the F1 local dynamics prior;
- executability/support constraints;

what multi-step latent trajectory should be followed?
```

F2 is the main new research object.

A method should only be called MPC-like if it performs a finite-horizon optimization and later, in closed loop:

```text
plan
execute a short prefix
observe/update
replan
```

Do not call one-step feed-forward refinement MPC.

## F3 — Task Executive

F3 answers:

```text
What is the current subgoal?
Is it complete?
What is the next subgoal?
```

F3 also maintains execution memory.

EXP_R1 must use an oracle F3 whenever valid ground-truth completion or task-boundary information exists.

Do not train learned F3 before latent path planning itself has been tested under oracle switching.

---

# 5. F3 waypoint memory and return

F3 must maintain a waypoint trace:

```text
T = {W_0, W_1, ..., W_K}
```

Waypoints must use the exact control/state fields present in the repository.

Do not guess:

```text
joint field names
gripper field names
TCP pose keys
state dictionary keys
controller-state paths
```

Audit source and data first.

At every subgoal switch, save:

```text
branch_checkpoint
```

The first `return` behavior is intentionally simple:

> **Stop forward task progression and replay or track the recorded waypoint sequence backward until the selected branch/start checkpoint is reached.**

The first claim should be limited to:

```text
robot execution-state return
```

Do not claim exact world-state reversal.

Possible variants, if supported by the actual controller interface:

```text
reverse raw-action replay
reverse joint-waypoint replay
reverse Cartesian waypoint tracking
MPC-guided waypoint tracking
```

Measure object/environment recovery separately.

---

# 6. EXP_R1 primary scientific question

EXP_R1 must answer:

> **Given an oracle start point, an oracle action-switch time, and an oracle or train-derived Action-B target region, can a trajectory planner connect Action A to Action B in the frozen latent space while preserving decoded-action continuity and empirical executability?**

Prefer a real ordered transition already present in continuous trajectories, for example a genuine `lift -> place` transition if confirmed by the repository data.

Use exact task identifiers recovered from data.

Do not invent names.

---

# 7. EXP_R1 required audits

Before scientific optimization, create:

```text
reports/EXP_R1_interface_audit.md
reports/EXP_R1_data_audit.md
reports/EXP_R1_frozen_manifest.json
reports/EXP_R1_preregistration.json
```

The interface audit must recover the exact:

```text
representation checkpoint identity
representation latent dimension
semantic/execution split
action-window length
latent stride/timing convention
decoder input/output interface
F1 exact inputs
F1 exact outputs
F1 timing convention
existing F2 exact inputs
existing F2 exact outputs
existing F2 refinement count
available continuous trajectory sources
annotation/task boundaries
available robot/state fields
available action fields
available success/completion fields
available waypoint/controller fields
```

If any identifier or field cannot be found exactly, inspect source/tests/artifacts until resolved.

Do not guess.

---

# 8. EXP_R1 dataset construction

Construct ordered path-planning cases from real continuous trajectories.

A planning case should contain, where scientifically valid:

```text
source trajectory/session ID
Action A identity
Action B identity
start latent z_A
Action-B goal set G_B
real hidden intermediate path for evaluation only
real future Action-B path for evaluation only
causal context available at planning issue time
```

The planner must not consume the hidden intermediate path.

Target-set construction must use train-only information.

Maintain source-session separation.

No held-out future state may leak into planning.

---

# 9. EXP_R1 broad method tournament

EXP_R1 must compare several fundamentally different path-planning mechanisms.

Do not spend the entire experiment tuning one model family.

## Class A — Non-planning controls

Attempt valid versions of:

```text
linear latent interpolation
spherical interpolation if geometrically meaningful
nearest-neighbor endpoint interpolation
F1 free rollout
existing old F2 refinement
best historical Wave28–Wave78 pointwise steering baseline
```

## Class B — Graph-based planning

Build a train-only graph over executable latent samples/prototypes.

Possible methods:

```text
kNN graph + Dijkstra
A*
task-conditioned graph
F1-weighted edge cost
decoder-continuity-weighted edge cost
```

Graph construction must preserve provenance and avoid test leakage.

## Class C — Direct trajectory optimization

Optimize latent waypoints directly.

Candidate costs:

```text
terminal target-set distance
F1 local-dynamics deviation
decoded-action continuity
decoded second-order smoothness
execution support / kNN density
decoder reconstruction/cycle diagnostic
path length
curvature
trust region
```

Do not blindly combine all terms.

Run structured ablations.

## Class D — Sampling / shooting planning

Where feasible, compare valid implementations of:

```text
random shooting
CEM
MPPI-style sampling
population-based latent trajectory search
```

Use only methods that can be correctly implemented under the actual interfaces.

## Class E — Differentiable planning

If F1 and decoder support gradient flow, test:

```text
gradient optimization over latent waypoint variables
gradient optimization over residual control variables
multiple-shooting formulation
soft dynamics penalties
hard/repair rollout variants
```

## Class F — Learned/amortized proposal

If justified by compute and data:

```text
small trajectory proposal network
goal-conditioned trajectory proposal
proposal + optimization refinement
```

This must remain multi-step and path-based, not pointwise steering.

## Class G — Hybrid global/local planner

Especially if Euclidean geometry is disconnected:

```text
graph path for global route
continuous optimization for local smoothing
```

---

# 10. F2 planning objective library

Implement a reusable cost library.

Generic objective:

```text
J =
    lambda_terminal * J_terminal
  + lambda_dyn      * J_dyn
  + lambda_cont     * J_cont
  + lambda_exec     * J_exec
  + lambda_support  * J_support
  + lambda_path     * J_path
  + lambda_curve    * J_curve
```

## J_terminal

Measures whether the final planned latent enters the Action-B-compatible target region.

Possible train-derived forms:

```text
nearest target-set distance
target-set Mahalanobis distance
semantic target-region score
retrieval target score
learned target-support critic
```

Exact held-out future endpoint information may only be used in an explicitly labeled oracle-endpoint diagnostic.

## J_dyn

Penalizes local transitions inconsistent with the exact existing F1 interface.

Conceptually:

```text
sum ||planned_next - F1(causal inputs)||^2
```

Do not assume a Markov form if F1 uses multiple history steps.

## J_cont

Penalizes decoded action discontinuity.

Measure at least:

```text
switch-time decoded action jump
first-order decoded action differences
```

## J_exec

Measures command-space validity / decodability using repository-supported diagnostics.

Possible signals:

```text
decoded continuous-action validity
gripper validity
decode/re-encode diagnostic
```

Do not assume decoder cycle alone is sufficient.

Historical evidence already showed that lower cycle residual can damage target identity.

## J_support

Discourages unsupported latent regions.

Possible train-derived measures:

```text
nearest-training distance
kNN radius
local density
train-only support classifier
```

Use:

```text
empirical executable latent support
```

unless a stronger geometric interpretation is directly established.

## J_path

Penalizes unnecessary trajectory length.

## J_curve

Penalizes excessive curvature / second-order changes.

---

# 11. Structured objective ablations

At minimum compare:

```text
terminal only

terminal + dynamics

terminal + continuity

terminal + support

terminal + dynamics + continuity

terminal + dynamics + support

terminal + continuity + support

terminal + dynamics + continuity + support

full objective
```

If new terms are introduced, perform targeted leave-one-out ablations.

Determine which constraints are genuinely necessary.

---

# 12. Planning horizons

Use multiple valid latent planning horizons.

Prefer a structured set such as:

```text
H = 2, 4, 8
```

or the closest horizons supported by the actual non-overlapping latent timing.

Report the physical duration represented by each horizon.

Do not introduce overlapping-window leakage.

---

# 13. Goal formulations

If data support them, compare:

```text
oracle exact endpoint
oracle/train-derived Action-B target set
language-derived Action-B target region
```

The exact endpoint is an upper-bound diagnostic.

The target-set case is the main planning problem.

Language-derived targeting is more realistic, but do not allow weak target grounding to obscure whether path planning itself works.

---

# 14. Required EXP_R1 metrics

For every method report:

```text
target-region arrival rate
endpoint semantic identity
trajectory latent error against hidden true path
decoded continuous-action error
switch-time decoded action jump
mean decoded first-order difference
decoded second-order smoothness
execution kNN/support distance
fraction of planned states beyond train-derived support threshold
F1 dynamics-consistency cost
path length
path curvature
optimization iterations
runtime
failure rate
non-finite rate
```

If physical rollout is available and scientifically valid, also report:

```text
task success
collision/safety events
robot state deviation
object state deviation
```

Do not equate offline decoded-action metrics with closed-loop task success.

---

# 15. EXP_R1 primary claim

The strongest intended first claim is:

> **A multi-step latent path planner can connect ordered atomic action regions more reliably than interpolation, free rollout, local refinement, and pointwise steering while preserving decoded-action continuity and empirical latent support.**

Only make this claim if evidence supports it.

Use:

```text
SUPPORTED
MIXED
NOT_SUPPORTED
NOT_TESTED
```

---

# 16. F1 policy in EXP_R1

Do not retrain F1 by default.

Use current F1 as a frozen local dynamics prior.

But test whether F1 actually helps the new planning problem.

Include:

```text
planner without F1 dynamics cost
planner with F1 dynamics cost
```

If F1 hurts planning, report it.

Historical local-prediction evidence remains valid, but usefulness as a planning prior is a new claim.

---

# 17. Existing F2 baseline

Retain the exact current F2 as a frozen baseline.

Do not modify it and still call it the historical baseline.

Give new planners distinct names, for example:

```text
F2_OLD_REFINER
F2_TRAJOPT
F2_CEM
F2_MPPI
F2_GRAPH
```

Use repository-appropriate identifiers while preserving clear provenance.

---

# 18. Oracle F3 in EXP_R1

Use oracle task boundaries/completion whenever valid.

EXP_R1 must isolate:

```text
latent path planning ability
```

not:

```text
progress recognition
```

If no valid oracle completion signal exists, document the exact limitation and construct the cleanest non-leaking offline protocol possible.

---

# 19. Learned F3 only after path-planning evidence

A later experiment may train F3 once path planning is viable.

Potential causal inputs, only if exactly available:

```text
long-horizon language
active atomic subgoal
frozen semantic latent
frozen execution latent
robot state
observation-derived causal state
causal history
```

Possible outputs:

```text
subgoal-completion probability
next-subgoal selection
branch checkpoint trigger
```

F3 should not generate continuous trajectories.

---

# 20. Long-horizon task sequencing

After path planning and F3 are sufficiently reliable, evaluate full long-horizon instructions.

Discover actual task chains from the repository/data.

Do not invent unsupported sequences.

Target logic:

```text
define ordered atomic subgoals
execute active subgoal
detect completion
switch target
plan latent transition
continue execution
```

The user should not need to issue every atomic action manually.

---

# 21. Return experiments

Once forward hierarchy works, add return.

At every branch:

```text
save branch checkpoint
record subsequent waypoints
```

When `return` is issued:

```text
stop forward task progression
select branch/start checkpoint
reverse the recorded waypoint reference
track/replay until the checkpoint is reached
```

Compare, if supported:

```text
raw-action reverse replay
joint-waypoint reverse replay
Cartesian waypoint reverse tracking
MPC-guided waypoint return
```

Primary metrics:

```text
final joint configuration error
final end-effector pose error
maximum tracking error
jerk / smoothness
collision/safety events
gripper-state recovery
```

Object/environment recovery must be reported separately.

Do not claim perfect environment reversal unless directly measured.

---

# 22. Mandatory report after every experiment

After every `EXP_R{id}`, create:

```text
reports/EXP_R{id}_report.md
```

Example:

```text
reports/EXP_R1_report.md
```

The report must be as detailed as reasonably possible.

It must contain:

```text
scientific question
previous bottleneck being targeted
repository commit
environment
data provenance
train/dev/held-out split
frozen checkpoint hashes
exact interfaces
methods attempted
architectures
planning algorithms
losses/costs
hyperparameter ranges
number of runs
valid runs
invalid runs
implementation failures
numerical failures
development results
held-out results
ablation results
statistical analysis
qualitative failure modes
figures/tables generated
strongest baselines
what succeeded
what failed
what was learned
claim decisions
remaining bottleneck
```

Preserve negative results.

Never overwrite an older report.

---

# 23. Mandatory next-experiment document

After every `EXP_R{id}`, also create:

```text
reports/next_exp_fromR{id}.md
```

Example:

```text
reports/next_exp_fromR1.md
```

This file must explain:

```text
what the current experiment established
what remains unresolved
most likely failure mechanism
what the next experiment should test
why it is scientifically justified
which modules must remain frozen
which modules may change
which baselines are required
which new methods should be attempted
success criteria
falsification criteria
```

Stay close to the main research direction whenever possible.

However, if the current method family clearly fails, Codex may propose a new method family.

Potential sources include:

```text
latent-space planning
trajectory optimization
MPC
CEM
MPPI
graph search
world-model planning
diffusion/flow planning
goal-conditioned control
hierarchical control
skill chaining
recent latent-action papers
recent robotics/control papers
official repositories
```

The next experiment must not merely change one scalar repeatedly.

Prefer broad mechanistic comparisons.

---

# 24. Broad experiment requirement

Each EXP_R experiment should, when computationally feasible, test multiple meaningful implementation directions.

Do not create a scientific sequence like:

```text
EXP_R2 = only change learning rate
EXP_R3 = only change one loss weight
EXP_R4 = only change batch size
```

unless debugging a concrete numerical failure.

A path-planning experiment should ideally compare several mechanisms such as:

```text
graph planning
gradient trajectory optimization
CEM
MPPI
proposal + refinement
```

under the same frozen representation.

A later F1 redesign experiment might compare:

```text
deterministic MLP
mixture-density dynamics
multi-hypothesis dynamics
goal-conditioned dynamics
controlled dynamics
```

rather than only changing hidden width.

---

# 25. Autonomous internet/literature research

Codex is authorized and encouraged to search the internet when:

```text
the current method fails
a planning algorithm is unclear
an implementation is unstable
a recent paper may solve the diagnosed mechanism
a library/API has changed
an official repository contains a useful implementation
```

Prefer primary sources:

```text
arXiv / conference papers
official project pages
official GitHub repositories
official documentation
```

Whenever external research is used, create or append:

```text
reports/EXP_R{id}_external_research.md
```

Record:

```text
search query
paper/repository title
URL
date accessed
paper year
repository commit/tag if applicable
specific borrowed idea
why it is relevant
how it was adapted
```

Do not copy a complete method blindly.

Adapt it to the exact repository interfaces and research question.

---

# 26. Problem-solving autonomy

Codex may independently:

```text
inspect source
write unit tests
fix implementation bugs
install compatible dependencies
read official documentation
search the web
change numerically equivalent implementations
reduce batch size
change optimizer implementation
add logging
add diagnostics
add plots
add analysis scripts
```

If a change modifies:

```text
scientific hypothesis
architecture
loss/cost
dataset
split
metric
claim threshold
```

after preregistration, move that change to the next EXP_R experiment.

Do not rescue a failed held-out result inside the same experiment.

---

# 27. Train/development/held-out discipline

Every EXP_R experiment must separate:

```text
TRAIN / DEVELOPMENT
```

from:

```text
HELD-OUT / FINAL EVALUATION
```

Broad exploration is allowed on train/development.

Before opening held-out:

```text
freeze final candidates
freeze metrics
freeze thresholds
freeze seeds
freeze manifests
```

Open held-out once.

If held-out fails:

```text
preserve result
diagnose
write reports/next_exp_fromR{id}.md
start EXP_R{id+1}
```

Never erase a negative result.

---

# 28. Automatic iteration loop

After completing `EXP_R{id}`:

```text
1. write reports/EXP_R{id}_report.md
2. write reports/next_exp_fromR{id}.md
3. decide whether SUCCESS has been reached
4. if not, generate the exact scientific plan for EXP_R{id+1}
5. preregister EXP_R{id+1}
6. execute EXP_R{id+1}
```

Continue autonomously until success or `EXP_R80`.

Do not wait for user confirmation between experiments.

---

# 29. Success stop

Stop early only when a defensible target system is achieved.

A strong success should include most of:

```text
latent path planning clearly beats pointwise steering/interpolation
F1 provides useful local dynamics information
F2 performs multi-step receding-horizon planning
planned paths remain continuous and empirically executable
multiple atomic transitions succeed
F3 sequences long-horizon subgoals
long-horizon instructions execute without manual atomic commands
waypoint memory supports reliable robot execution-state return
major ablations support the roles of F1/F2/F3
held-out or prospective evidence supports the claims
```

On success, generate:

```text
FINAL_R_METHOD.md
FINAL_R_PAPER_STORY.md
FINAL_R_ABLATION_PLAN.md
FINAL_R_FAILURE_ANALYSIS.md
FINAL_R_NEXT_CLOSED_LOOP_PLAN.md
```

Then stop.

---

# 30. EXP_R80 stop

If `EXP_R80` completes without success:

```text
do not start EXP_R81
```

Generate:

```text
FINAL_R80_RESEARCH_SUMMARY.md
FINAL_R80_FAILURE_TAXONOMY.md
FINAL_R80_SUPPORTED_CLAIMS.md
FINAL_R80_RECOMMENDED_DIRECTION.md
```

The conclusion may be negative.

Possible final bottlenecks include:

```text
representation lacks path connectivity
F1 is insufficient as a planning prior
state feedback is missing
latent timing is inappropriate
decoder interface is insufficient
controlled/state-action latent is required
```

Do not force a positive conclusion.

---

# 31. Questions every EXP_R report must answer

Every `reports/EXP_R{id}_report.md` must answer:

```text
What exact scientific hypothesis was tested?
What previous result motivated it?
What was frozen?
What was retrained or optimized?
What exact data were used?
Was any future leakage possible?
What exact F1 interface was used?
What exact F2/planner interface was used?
Was F3 oracle or learned?
What endpoint/goal formulation was used?
What planning algorithms were compared?
What costs/losses were compared?
What horizons were compared?
Which baselines were strongest?
Did planning beat linear interpolation?
Did planning beat F1 rollout?
Did planning beat old F2 refinement?
Did planning beat best pointwise steering?
Did F1 dynamics cost help?
Did continuity cost help?
Did support/executability cost help?
Did target-set planning work without exact endpoint leakage?
Did language-derived targeting work?
Did path length/curvature become pathological?
Did any method exploit unsupported latent gaps?
Were decoded actions continuous?
Were endpoints semantically correct?
Were endpoints motor-valid?
What failed most often?
Which module is now the bottleneck?
Is representation connectivity supported?
Is F1 useful as a local dynamics prior?
Is F2 legitimately MPC-like yet?
Is F3 justified yet?
Is closed-loop execution justified yet?
Is return justified yet?
Was external research used?
What was borrowed?
Was held-out opened only after freeze?
Which claim is supported?
Which claim is unsupported?
What should EXP_R{id+1} do?
```

---

# 32. EXP_R1 minimum comparison table

At minimum include valid implementations of:

```text
Ground-truth hidden path        evaluation reference only
Linear interpolation
F1 free rollout
Old F2 refinement
Best historical pointwise steering
kNN/Dijkstra or equivalent graph planner
Trajectory optimization: terminal only
Trajectory optimization: terminal + F1
Trajectory optimization: terminal + continuity
Trajectory optimization: terminal + support
Trajectory optimization: terminal + F1 + continuity + support
Sampling planner: CEM / MPPI / equivalent if feasible
Best hybrid method
```

If a method is impossible under the actual interface, document the reason.

Do not fabricate results.

---

# 33. EXP_R1 success indicators

EXP_R1 is promising if at least one true path-planning method shows a consistent Pareto improvement over non-planning baselines across:

```text
target-region arrival
decoded continuity
empirical latent support
hidden-path similarity
```

Do not require complete real-world long-horizon success in EXP_R1.

The first milestone is:

> **multi-step path structure exists and can be exploited.**

---

# 34. Research guardrail

Do not drift back into:

```text
another low-dimensional force-field sweep
another residual-adapter sweep
another static goal-core attraction method
another decoder-cycle-only rescue
```

unless a later experiment produces a specific new theoretical justification and compares it against the path-planning formulation.

The primary research line is now:

```text
representation
    ->
local dynamics
    ->
latent path planning
    ->
hierarchical subgoal navigation
    ->
waypoint-based return
```

---

# 35. First actions

Begin `EXP_R1` now.

Proceed in this order:

```text
read revised direction
read historical representation/F1/F2/Wave21–78 evidence
audit exact repository interfaces
freeze representation and F1 hashes
identify real ordered atomic transitions
construct train/dev/held-out path-planning cases
implement non-planning controls
implement graph-based planning
implement differentiable trajectory optimization
implement at least one sampling planner if feasible
implement structured cost library
run development tournament
construct Pareto analysis
freeze final candidates
open held-out once
write detailed EXP_R1 report
write next_exp_fromR1.md
decide SUCCESS or EXP_R2
```

The final mature system should support:

```text
long-horizon instruction
    ->
F3 selects Action A
    ->
F1 predicts local A evolution
    ->
F2 maintains feasible execution
    ->
F3 detects A completion
    ->
F3 switches target to Action B
    ->
F2 plans a continuous latent path toward B
    ->
F1 resumes local dynamics prediction inside B
```

For `return`:

```text
F3 loads the selected execution trace
    ->
reverse waypoint reference
    ->
F2/controller tracks the trace
    ->
robot returns to the recorded execution checkpoint
```

Begin `EXP_R1`.
