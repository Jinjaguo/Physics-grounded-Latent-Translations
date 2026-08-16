# EXP_R9+ Autonomous Codex Research Program
## Closed-Loop Latent MPC, Long-Horizon Task Navigation, and Waypoint Return

# 0. Program status and naming

You are Codex operating inside the existing **Actions as Coordinates** research repository.

The previous autonomous stage successfully stopped at `EXP_R8`. Preserve its result: the selected method was `repair_late_0.75`, with held-out arrival `1.0000`, decoded first difference `1.0375`, and hidden-path MSE `0.9743`. The EXP_R8 claim was `SUPPORTED`.

This document starts a new autonomous research stage from:

```text
EXP_R9
```

and allows at most 50 new experiments:

```text
EXP_R9 ... EXP_R58
```

Stop immediately when the full target system is scientifically achieved. Never start `EXP_R59`.

Do not return to Wave numbering. Do not delete, overwrite, or reinterpret EXP_R1–EXP_R8 history.

---

# 1. Read the repository before changing anything

Before implementing EXP_R9, read the actual repository artifacts for:

```text
EXP_R1 ... EXP_R8 reports
EXP_R8 claim decision
EXP_R8 selected method and implementation
representation readiness / frozen representation
F1 source and reports
historical F2 source and reports
R4–R8 proposal / repair code
continuous-play datasets
annotation boundaries
robot/controller interfaces
state/action schemas
tests and manifests
```

Also read the current post-Wave78 research-direction document if present.

Repository source, machine-readable artifacts, checkpoint manifests, and dataset manifests are authoritative.

Do not guess task IDs, checkpoint names, latent dimensions, state keys, action keys, controller fields, completion labels, waypoint fields, or timing conventions.

---

# 2. Preserve the EXP_R8 scientific conclusion

EXP_R8 supports the following limited but important claim:

> **Action A to Action B can be modeled more successfully as a multi-step latent path than as a pointwise displacement.**

Do not reopen EXP_R8 merely to search for a numerically better repair coefficient.

Do not restart pointwise force-field / residual-adapter sweeps as the main line.

Do not describe EXP_R8 as closed-loop MPC. EXP_R8 currently supports offline/open-loop multi-step latent path construction. The new stage must test closed-loop control, automatic task progression, long-horizon instruction following, and waypoint return.

---

# 3. Final target system

The final system must satisfy four capabilities.

## Goal 1 — Stable atomic action execution

Atomic actions must execute stably before and during long-horizon composition.

Current representation, F1, and historical F2 already provide the foundation.

F1 is responsible for local progression of the currently active atomic action. Historical F2 provides evidence that local iterative refinement is useful.

If the robot is still executing `lift`, and `lift` is incomplete, the system should continue executing `lift`. A future `place` subgoal must not pull the robot away prematurely.

The system should preserve:

```text
current-action identity
decoded action continuity
empirical executability
local progression
```

## Goal 2 — Action A to Action B through a latent path

Transitions must occur through a multi-step latent path:

```text
z_0 -> z_1 -> ... -> z_H
```

with:

```text
z_0 = current executable latent
z_H in target Action-B-compatible region
```

Intermediate states are selected by the planner/controller rather than manually specified.

Preserve the working hypothesis:

> **Action transitions are paths, not displacements.**

## Goal 3 — Upgrade F2 into true closed-loop latent MPC

The new F2 must perform a genuine receding-horizon loop:

```text
observe current real execution state
    ->
encode / recover current latent
    ->
plan H latent steps
    ->
execute only a short prefix
    ->
observe actual resulting state
    ->
re-encode
    ->
replan
    ->
repeat
```

The core loop is:

```text
plan H steps -> execute short prefix -> observe -> re-encode -> replan
```

Do not call a method MPC if it generates one full path and executes it completely open-loop.

## Goal 4 — F3 controls long-horizon task progression

F3 must become the actual high-level Task Executive.

For a long-horizon instruction such as:

```text
lift the blue block, then place it in the slider
```

F3 should maintain an ordered subgoal sequence and active subgoal. While `lift` is incomplete, it keeps `lift` active. Once `lift` completes, F3 switches the target to `place` and hands the new target region/reference to F2.

The user should not need to issue the second atomic instruction manually.

---

# 4. Module responsibilities

## F1 — Local latent dynamics

F1 answers:

> **If the current atomic behavior continues naturally, where should the latent go next?**

Conceptually:

```text
z_next_nominal = F1(causal latent history, causal context)
```

F1 should not initially solve:

```text
long-horizon task decomposition
subgoal completion
lift -> place switching
global path planning
return
```

Keep the accepted F1 frozen at the start of the new stage.

Only retrain or redesign F1 later if a specific experiment demonstrates that F1 itself is the bottleneck for closed-loop control.

If F1 is later modified, retain the historical frozen F1 as a baseline.

## F2 — Latent trajectory controller / MPC

F2 answers:

> **Given the current execution state, active target, and F1 local-dynamics information, what feasible multi-step latent path should be executed now?**

During normal atomic execution:

```text
F1 gives nominal local continuation
F2 keeps the actual latent/action trajectory smooth and executable
```

During a subgoal transition:

```text
F3 changes the target
F2 increases goal-directed authority
F2 plans a feasible path into the new action region
```

Once the new action region is reached:

```text
F1 again provides local nominal progression
F2 continues receding-horizon control
```

Treat `repair_late_0.75` as an important R8 baseline and structural clue, not as a universal final controller.

Explicitly test the hypothesis:

> **preserve local dynamics early; enforce the new destination later.**

## F3 — Task Executive + execution memory

F3 is responsible for:

```text
long-horizon subgoal sequence maintenance
current subgoal tracking
subgoal completion detection
subgoal switching
target-region handoff to F2
branch checkpoint management
waypoint memory
return command handling
```

F3 must not generate the continuous latent trajectory itself.

---

# 5. Oracle F3 before learned F3

Do not begin EXP_R9 by changing both F2 and F3.

First isolate closed-loop F2 using valid oracle subgoal completion / switch timing where available.

Required progression:

```text
closed-loop F2 with oracle F3
    ->
robust closed-loop transitions
    ->
learned F3
    ->
full automatic long-horizon sequencing
```

If F2 fails under oracle F3, do not blame F3.

---

# 6. F3 waypoint memory and return

F3 must maintain an execution trace:

```text
T = {W_0, W_1, ..., W_K}
```

Waypoints must use the exact minimal robot/controller state available in the repository and required for reliable tracking/replay. Audit the interface; do not guess fields.

At every subgoal switch or branch, save:

```text
branch_checkpoint
```

The first `return` behavior should deliberately remain simple.

When `return` is issued, F3 should:

```text
stop forward task progression
select the most recent branch checkpoint or designated start checkpoint
load the stored waypoint trace
reverse the trace
send the reversed reference to the return controller
```

Possible variants, only if supported by the actual interface:

```text
reverse raw-action replay
reverse joint-waypoint replay
reverse Cartesian waypoint tracking
MPC-guided waypoint tracking
```

The first claim must be limited to:

> **The system can return the robot to a previously recorded execution state.**

Do not claim exact world-state reversal. Measure object/environment restoration separately.

---

# 7. EXP_R9 primary question

EXP_R9 begins the transition from R8 open-loop path planning to closed-loop control.

Primary question:

> **Does receding-horizon replanning preserve or improve the R8 transition result when the system repeatedly updates from the actually observed / re-encoded state instead of executing one complete open-loop latent path?**

EXP_R9 should keep:

```text
representation frozen
decoder frozen
F1 frozen
historical old F2 frozen as baseline
R8 path planner available as baseline/initializer
oracle F3
```

The primary new object is the closed-loop F2 design.

---

# 8. EXP_R9 required interface audit

Before running control experiments, create:

```text
reports/EXP_R9_interface_audit.md
reports/EXP_R9_closed_loop_preregistration.json
reports/EXP_R9_frozen_manifest.json
```

Recover exactly:

```text
how an executed robot/action state is observed
how that state is converted back to the representation input
how the current latent is re-encoded after execution
how F1 consumes causal history
how the R8 planner is initialized
how decoder outputs are executed or replayed
what constitutes one executable prefix
what timing one latent step represents
what oracle completion/switch signal is available
what simulator or replay interface is available
```

If actual closed-loop execution cannot be run, identify the strongest causally valid replay/closed-loop surrogate and document the limitation explicitly.

---

# 9. EXP_R9 broad controller tournament

Do not test only one receding-horizon implementation.

At minimum, where interfaces permit, compare:

```text
R8 open-loop full-path baseline
R8 receding-horizon: plan H, execute 1 latent step, replan
R8 receding-horizon: plan H, execute short action prefix, replan
warm-start receding-horizon using previous path tail
cold-start receding-horizon
F1-only closed-loop rollout
old-F2 closed-loop local refinement
trajectory optimization initialized by R8 proposal
graph/global-route + local MPC
CEM / MPPI / sampling MPC if feasible
```

Do not reduce EXP_R9 to a coefficient sweep.

---

# 10. Planning horizon and execution-prefix study

Study both:

```text
planning horizon H
execution prefix length P
```

because receding-horizon behavior depends on both.

Use only timing choices supported by the actual latent/action representation.

Possible conceptual combinations include:

```text
plan H=4, execute P=1
plan H=4, execute P=2
plan H=8, execute P=1
plan H=8, execute P=2
```

Do not invent unsupported timing.

---

# 11. Closed-loop F2 objective

Implement a structured controller objective. A generic form is:

```text
J =
    lambda_goal(k) * J_goal
  + lambda_dyn(k) * J_dyn
  + lambda_cont    * J_cont
  + lambda_exec    * J_exec
  + lambda_support * J_support
  + lambda_path    * J_path
  + lambda_curve   * J_curve
```

The new key question is whether goal and dynamics authority should vary with horizon position.

Test the R8-inspired family:

```text
early horizon:
    stronger local-dynamics preservation

late horizon:
    stronger target attraction / terminal capture
```

Compare, when feasible:

```text
fixed weighting
linear schedule
piecewise early/late schedule
adaptive schedule
confidence-dependent schedule
distance-to-target schedule
```

Do not assume the fixed historical value `0.75` is universal.

---

# 12. EXP_R9 metrics

At minimum report:

```text
target-region arrival
number of replans
time / steps to target
decoded first difference
decoded second-order smoothness
maximum switch-time jump
hidden-path MSE where reference exists
F1 consistency
empirical support distance
fraction outside support threshold
cumulative path length
path curvature
replanning instability
oscillation count
target overshoot
runtime per replan
non-finite rate
```

If simulator or physical rollout is available, also report:

```text
atomic task success
transition task success
collision/safety events
robot-state error
object-state error
```

Do not equate offline latent metrics with physical task success.

---

# 13. When F2 is allowed to be called MPC

Only call the new F2 **Latent MPC** after the implementation actually demonstrates:

```text
finite-horizon planning
short-prefix execution
feedback from the actual resulting state
re-encoding / state update
replanning
```

If feedback is simulated only through an offline surrogate, label it accurately.

---

# 14. Gate for moving to learned F3

Do not train F3 just because EXP_R9 finishes.

Move to learned F3 only after closed-loop F2 under oracle switching is sufficiently stable.

Preregister a criterion including most of:

```text
reliable target arrival
acceptable decoded continuity
no systematic oscillation
no severe support violation
consistent results across multiple ordered action pairs
```

Freeze the criterion before held-out evaluation.

---

# 15. Learned F3 experiments

Once closed-loop F2 is supported, start a later EXP_R experiment focused on F3.

F3 should predict at least:

```text
p(subgoal_complete)
```

and, if needed by the actual task representation:

```text
next_subgoal
```

Possible causal inputs, only if actually available, include:

```text
active atomic subgoal
long-horizon language
frozen semantic latent
frozen execution latent
robot state
causal observation features
latent/action history
F1 prediction error
F2 target distance / progress
```

Do not expose hidden future labels at inference time.

---

# 16. F3 evaluation

Evaluate both classifier-level and system-level behavior.

Metrics may include:

```text
completion AUROC/AUPRC
balanced accuracy
early-switch rate
late-switch rate
mean switch delay
false-completion rate
missed-completion rate
task-pair generalization
source-session generalization
```

More importantly, measure:

```text
whether early switching corrupts unfinished atomic actions
whether late switching prevents transition
whether learned F3 approaches oracle-F3 system performance
```

The key safety property is:

> **Future subgoals must not prematurely corrupt an unfinished current action.**

---

# 17. Long-horizon instruction experiments

After F2 and F3 are individually supported, evaluate complete long-horizon instructions.

Use real supported task sequences discovered from the dataset/repository.

Do not invent unsupported chains.

The user provides one long-horizon instruction. The system should autonomously:

```text
maintain ordered atomic subgoals
execute current atomic action
detect completion
switch target
plan the latent transition
execute closed-loop
continue until final completion
```

Start with two-step tasks. Expand to longer chains only after two-step tasks are reliable.

---

# 18. Long-horizon baselines

Where valid, compare against:

```text
manual oracle switching
fixed-time switching
annotation-boundary oracle
no F1
no F2 planning
old F2 only
linear transition
R8 open-loop transition
new F2 closed-loop MPC
learned F3 + new F2
```

If a monolithic long-horizon baseline exists in the repository, include it.

---

# 19. Waypoint return experiments

After forward long-horizon execution becomes reliable, evaluate return.

At minimum test:

```text
return to most recent branch checkpoint
return to long-horizon task start
```

if supported.

Compare, where possible:

```text
reverse action replay
reverse joint-waypoint replay
tracking-controller return
MPC-guided waypoint return
```

Measure:

```text
joint configuration error
end-effector pose error
gripper-state error
maximum tracking error
jerk / smoothness
collision/safety events
return completion rate
object-state error separately
```

The primary claim remains robot execution-state return.

---

# 20. Mandatory report after every experiment

After every `EXP_R{id}`, create:

```text
reports/EXP_R{id}_report.md
```

The first new report is:

```text
reports/EXP_R9_report.md
```

Do not create another `EXP_R8_report.md`.

Each report must be as detailed as reasonably possible and include:

```text
scientific question
motivation from previous EXP
repository commit
environment and hardware
data provenance
train/dev/held-out split
frozen checkpoint hashes
exact F1/F2/F3 interfaces
what remained frozen
what changed
methods attempted
architectures
planning/control algorithms
losses/costs
weight schedules
horizons
execution prefix lengths
hyperparameter ranges
number of runs
valid/invalid runs
implementation failures
numerical failures
development results
held-out results
simulator/physical results
ablations
statistical analysis
failure modes
plots/tables
claim decision
remaining bottleneck
```

Never overwrite older reports.

Preserve negative results.

---

# 21. Mandatory next-experiment document

After every `EXP_R{id}`, also create:

```text
reports/next_exp_fromR{id}.md
```

For EXP_R9:

```text
reports/next_exp_fromR9.md
```

This file must explain:

```text
what the experiment established
what failed
what remains unresolved
most likely failure mechanism
what the next experiment should test
why it is scientifically justified
which modules remain frozen
which modules may change
which baselines remain mandatory
which new methods should be attempted
success criteria
falsification criteria
```

Stay close to the main research direction whenever possible.

If the current method family clearly cannot solve the diagnosed problem, Codex is authorized to propose and implement a new method family.

---

# 22. Broad-experiment requirement

Each EXP_R experiment should, when computationally feasible, compare multiple meaningful approaches.

Do not create a sequence like:

```text
EXP_R10 = only change lambda
EXP_R11 = only change learning rate
EXP_R12 = only change batch size
```

unless debugging a specific numerical failure.

Examples of broad F2 experiments:

```text
warm-start trajectory optimization
cold-start trajectory optimization
CEM
MPPI
graph + local optimization
adaptive early/late weighting
```

Examples of broad F3 experiments:

```text
simple classifier
temporal MLP
GRU/Transformer history model
progress regression
hazard/termination model
state + latent fusion
```

Examples of broad return experiments:

```text
raw replay
waypoint replay
tracking controller
MPC tracking
```

---

# 23. Autonomous literature and internet research

Codex is authorized and encouraged to search the internet when needed.

Search when:

```text
closed-loop planning is unstable
MPC formulation is unclear
controller oscillation appears
rollout becomes numerically unstable
F3 completion detection fails
waypoint replay is unsafe
a recent paper addresses the diagnosed mechanism
an official implementation improves correctness
```

Prefer primary sources:

```text
conference papers
arXiv
official project pages
official GitHub repositories
official documentation
```

For every EXP using external research, create or append:

```text
reports/EXP_R{id}_external_research.md
```

Record:

```text
search query
paper/repository title
URL
access date
paper year
repository commit/tag if applicable
specific borrowed idea
why it is relevant
how it was adapted
```

Relevant areas include, but are not limited to:

```text
latent MPC
TD-MPC / TD-MPC2 style planning
CEM
MPPI
iLQR-style trajectory optimization
shooting / multiple-shooting methods
hierarchical MPC
goal-conditioned control
skill termination / options
hierarchical RL
latent world-model control
task-progress prediction
waypoint tracking
trajectory replay
learning from demonstrations
```

Do not copy methods blindly. Adapt them to the actual project interfaces and evidence.

---

# 24. Problem-solving autonomy

Codex may independently:

```text
inspect source code
write tests
fix implementation bugs
install compatible dependencies
search official documentation
search literature
change numerically equivalent implementations
reduce batch size
change optimizer implementation
add diagnostics
add logging
add plots
add analysis scripts
```

If a proposed change modifies the scientific hypothesis, architecture, dataset, split, metric, or claim threshold after preregistration, move it to the next EXP_R experiment.

Do not rescue a failed held-out result inside the same experiment.

---

# 25. Do not retrain everything simultaneously

Preserve modular attribution.

Default progression:

```text
Stage A:
freeze representation
freeze F1
oracle F3
develop closed-loop F2

Stage B:
freeze successful F2
train/evaluate F3

Stage C:
integrate F2 + learned F3

Stage D:
add waypoint return

Stage E:
full long-horizon system
```

Only reopen F1 if evidence specifically identifies F1 as the bottleneck.

Only reopen the representation if evidence specifically identifies latent connectivity or decoder support as the bottleneck.

---

# 26. Held-out discipline

Every EXP_R experiment must separate:

```text
TRAIN / DEVELOPMENT
```

from:

```text
HELD-OUT / FINAL EVALUATION
```

Broad exploration is allowed on development.

Before opening held-out:

```text
freeze final candidates
freeze metrics
freeze thresholds
freeze seeds
freeze manifests
freeze checkpoint hashes
```

Open held-out once.

If held-out fails:

```text
preserve failure
write EXP report
write next_exp document
start the next experiment
```

Never erase or retroactively modify a negative result.

---

# 27. Automatic iteration loop

After every EXP_R experiment:

```text
1. finish preregistered development analyses
2. freeze candidate(s)
3. evaluate held-out once
4. write reports/EXP_R{id}_report.md
5. write reports/next_exp_fromR{id}.md
6. decide whether overall SUCCESS is reached
7. if not, preregister EXP_R{id+1}
8. execute EXP_R{id+1}
```

Continue autonomously from EXP_R9 through at most EXP_R58.

Do not ask for user confirmation between experiments.

---

# 28. Overall success condition

Do not stop merely because one offline latent metric improves.

The new stage reaches `SUCCESS` only when a defensible version of the full target system is achieved.

A strong success should include most of:

```text
1. atomic actions remain stable until their subgoal is complete

2. Action A -> Action B transition uses a multi-step latent path

3. F2 runs in true receding-horizon closed loop

4. closed-loop F2 preserves or improves the successful R8 behavior

5. multiple ordered action transitions work

6. F3 reliably detects subgoal completion and switches automatically

7. one long-horizon instruction executes multiple atomic subgoals without manual intermediate commands

8. future subgoals do not prematurely corrupt unfinished current actions

9. F3 maintains waypoint execution memory

10. return restores the robot to a recorded execution checkpoint with low robot-state error

11. ablations support distinct roles for F1, F2, and F3

12. held-out or prospective results support the final claims
```

The final claim must remain conservative and match measured capability.

---

# 29. Final target demonstration

The mature system should support a command such as:

```text
lift the blue block, then place it in the slider
```

Desired behavior:

```text
F3 sets current subgoal = lift

F1 predicts local lift continuation

F2 performs stable closed-loop latent control

F3 detects lift completion

F3 switches current subgoal = place

F2 plans a latent path toward the place-compatible region

F2 executes only a short prefix

system observes the actual state

system re-encodes

F2 replans

repeat until place succeeds
```

For:

```text
return
```

expected behavior is:

```text
F3 selects the relevant branch checkpoint
F3 loads the stored waypoint trace
trace is reversed
F2/controller tracks the reverse reference
robot returns to the recorded execution state
```

---

# 30. Paper-level target

The final paper should aim to support:

> **Actions are represented as language-addressable coordinates. F1 models how the current coordinate trajectory naturally evolves. F2 performs closed-loop trajectory optimization to move between action regions. F3 navigates long-horizon subgoals and stores execution memory for lightweight return.**

Core transition claim:

> **Action transitions are paths, not displacements.**

Broader control claim:

> **A language-grounded action representation can become a navigable control space when local dynamics, trajectory control, and task-level navigation are separated.**

---

# 31. Failure interpretations

If R8 open-loop succeeds but all closed-loop F2 variants fail:

```text
state re-encoding, feedback mismatch, or model mismatch is likely the bottleneck
```

If closed-loop F2 succeeds with oracle F3 but learned F3 fails:

```text
subgoal completion estimation is the bottleneck
```

If F3 detects completion correctly but some action pairs fail:

```text
pair-specific latent connectivity or local dynamics may be the bottleneck
```

If return restores robot configuration but not object state:

```text
claim execution-state return only
```

If atomic execution degrades merely because future subgoals are present:

```text
task-conditioning leakage / premature switching is the bottleneck
```

If F1 consistently hurts closed-loop planning:

```text
preserve its historical local-prediction claim, but stop using it as a control prior
```

If the frozen representation blocks closed-loop transitions across many pairs:

```text
only then consider a new controlled/state-action representation in a later EXP
```

---

# 32. Final stop rule

Continue from:

```text
EXP_R9
```

through at most:

```text
EXP_R58
```

Stop early if the full target system is achieved.

If EXP_R58 finishes without full success:

```text
do not start EXP_R59
```

Generate:

```text
FINAL_R9_R58_RESEARCH_SUMMARY.md
FINAL_R9_R58_FAILURE_TAXONOMY.md
FINAL_R9_R58_SUPPORTED_CLAIMS.md
FINAL_R9_R58_BEST_SYSTEM.md
FINAL_R9_R58_RECOMMENDED_NEXT_DIRECTION.md
```

Do not force a positive conclusion.

---

# 33. Required final artifacts on success

If the target system succeeds before EXP_R58, generate:

```text
FINAL_HIERARCHICAL_LATENT_CONTROL_METHOD.md
FINAL_F1_F2_F3_SYSTEM_DESCRIPTION.md
FINAL_CLOSED_LOOP_MPC_RESULTS.md
FINAL_LONG_HORIZON_RESULTS.md
FINAL_RETURN_RESULTS.md
FINAL_ABLATION_PLAN.md
FINAL_PAPER_STORY.md
FINAL_LIMITATIONS.md
```

Then stop autonomous experimentation.

---

# 34. Start EXP_R9 now

Begin EXP_R9 immediately.

First:

```text
audit exact closed-loop interfaces
recover exact R8 planner
recover exact F1 and historical F2 interfaces
identify how the executed robot state can be re-encoded
identify valid oracle subgoal boundaries
preregister multiple closed-loop F2 variants
```

Then run the EXP_R9 development tournament.

The first milestone of the new stage is:

> **Convert the successful EXP_R8 open-loop path planner into a true receding-horizon latent controller without losing arrival, continuity, or path realism.**

Begin.
