# Actions as Coordinates After Wave28–Wave78
## From Pointwise Latent Steering to Hierarchical Latent Path Planning

### Purpose

This document defines the revised scientific direction of the **Actions as Coordinates** project after Wave28–Wave78. It preserves the successful representation and dynamics results, while changing the control formulation that failed during the low-dimensional force-field program.

The central shift is:

> **Stop treating action retargeting as pointwise latent displacement. Start treating it as hierarchical path planning in a learned action-coordinate space.**

The previous program tested whether a frozen, language-addressable action latent could be made directly editable using low-dimensional force fields, residual adapters, local bases, decoder-aware projections, state conditioning, gating, cycle constraints, and related mechanisms. Those experiments repeatedly failed to achieve strong retargeting, continuity, and endpoint identity at the same time.

The scientific lesson is:

> **Language addressability, local predictability, and direct editability are distinct properties.**

The new paper should therefore ask whether a language-grounded action latent can function as a **navigable control coordinate system** when local dynamics, continuous trajectory planning, and high-level subgoal switching are separated.

---

# 1. What remains scientifically valid

The frozen representation remains valuable and should not be retrained merely because the later steering program failed.

The representation provides a continuous action coordinate with two key properties:

- language addressability;
- motor fidelity / decodability.

Its correct interpretation is:

> **The representation tells us where meaningful action regions are. It does not automatically tell us how to travel between them.**

The earlier F1/F2 dynamics results also remain useful.

F1 should be interpreted as a **learned local latent dynamics predictor**: given causal latent history/context, it predicts where the current action trajectory naturally evolves next.

The existing F2 should be interpreted as **iterative local refinement**: it improves F1 proposals and stabilizes local rollout under the previously tested setting.

The existing F2 result does **not** prove:

- Action A can transition to Action B;
- arbitrary endpoints are connectable;
- refinement performs planning;
- refinement is MPC;
- semantic goal switching is solved.

The earlier language-retargeting results also remain valid: changing the next-goal language can causally redirect predicted latent dynamics, including execution-space coordinates. This shows that language contains information about different future behavior.

The correct interpretation is:

> **Language can select different future behavior, but direct pointwise steering is not sufficient to realize a continuous executable transition.**

---

# 2. What Wave28–Wave78 established

Wave28–Wave78 explored a broad family of pointwise or locally parameterized retargeting mechanisms, including:

- 1D / 2D / 4D / 8D control spaces;
- random, PCA, learned, and state-dependent bases;
- static and dynamic residual fields;
- attractor-style updates;
- state-conditioned and nonlinear fields;
- gated intervention;
- decoder-aware caps;
- trust regions;
- semantic anchors;
- execution anchors;
- cycle and return objectives;
- phase/contact gating;
- decoder-Jacobian directions;
- tangent-space directions;
- multi-step consistency;
- full-rank controls;
- mixture experts;
- multiple loss combinations.

Different mechanism families repeatedly converged on the same failure pattern:

> stronger retargeting damages continuity / action identity, while stronger preservation suppresses retargeting.

Wave31 is especially informative because a learned intervention gate effectively reduced intervention magnitude toward near-zero. This indicates that preserving the original decoded behavior and producing a strong new target direction were structurally in conflict under the pointwise intervention formulation.

Wave34 correctly triggered a representation-level stop for continued adapter stacking.

Wave40 is also informative: separating semantic and execution branches produced one of the clearest execution-redirect improvements, but continuity remained poor. This supports the conclusion that semantic task switching and continuous execution should not be forced into one residual displacement.

The post-Wave78 interpretation should therefore be:

> **Direct latent steering is the wrong control formulation for this representation. The failure does not invalidate the action coordinates. It shows that action-space editability requires trajectory structure, not only target displacement.**

A concise paper statement is:

> **Language addressability is a property of representation organization; editability is a property of transition structure.**

---

# 3. New central hypothesis

The new hypothesis is:

> **Atomic action regions in a language-addressable latent space can be connected by feasible latent trajectories, provided that local latent dynamics, execution feasibility, and high-level subgoal switching are modeled as separate control functions.**

This replaces the old hypothesis that a small residual or low-dimensional field can directly move Action A toward Action B.

The new hierarchy is:

```text
F1 — Local Latent Dynamics
F2 — Latent Trajectory Controller / MPC-like Optimizer
F3 — Task Navigator + Execution Memory
```

The frozen representation remains the coordinate system.

The frozen decoder remains the interface from latent coordinates to robot actions.

---

# 4. F1: Local Latent Dynamics

F1 answers one deliberately local question:

> **Given the current action trajectory, where would the latent naturally move next if the current atomic behavior continues?**

Conceptually:

```text
z_next_nominal = F1(causal latent history, causal context)
```

F1 is not responsible for:

- long-horizon task decomposition;
- deciding that `lift` should become `place`;
- return;
- global route planning.

If the robot is still executing `lift` and the lift subgoal is incomplete, the correct F1 behavior is to continue predicting the local lift trajectory.

The existing F1 should initially remain frozen and act as a **local dynamics prior**.

A later experiment may retrain or extend F1 only if new evidence shows that the current local model is insufficient near subgoal boundaries or controlled branching states.

---

# 5. F2: From Refinement to Latent Trajectory Control

The current F2 is a useful starting point but should not yet be called MPC.

Current behavior is approximately:

```text
F1 one-step proposal
    ↓
iterative learned correction
    ↓
refined next latent
```

This supports:

> **Iterative correction of a local latent proposal is beneficial.**

The new F2 should solve a stronger problem:

> **Given the current latent, a target action region, and a local dynamics prior, find a feasible multi-step latent trajectory connecting the current state to the target region.**

Instead of optimizing one next latent, define:

```text
Z = {z_1, z_2, ..., z_H}
```

and optimize the entire path.

A general objective is:

```text
J =
    lambda_terminal * J_terminal
  + lambda_dyn      * J_dynamics
  + lambda_cont     * J_continuity
  + lambda_exec     * J_executability
  + lambda_support  * J_data_support
  + lambda_path     * J_path
  + lambda_curve    * J_curvature
```

`J_terminal` measures whether the path reaches an Action-B-compatible region.

`J_dynamics` measures consistency with F1 local dynamics.

`J_continuity` penalizes large differences between consecutive decoded actions.

`J_executability` penalizes latent states that decode into invalid or implausible commands.

`J_data_support` discourages trajectories through unsupported latent regions.

`J_path` and `J_curvature` prevent unnecessarily long or sharply changing latent trajectories.

The key design principle is:

> **Do not manually prescribe the intermediate states. Specify the start, the target region, and the constraints that define a feasible path.**

---

# 6. Start and target

Do not assume Action B is one unique latent point.

Use:

```text
start = current executable latent z_A
goal  = Action-B-compatible terminal region G_B
```

The planner solves:

```text
z_A -> z_1 -> z_2 -> ... -> z_H in G_B
```

The endpoint may be selected by the planner.

This avoids repeating the failed static goal-core attraction assumption.

---

# 7. F3: Task Navigator

F3 handles discrete long-horizon logic.

For a command such as:

```text
lift the blue block, then place it in the slider
```

F3 maintains:

```text
lift_blue_block_slider
    ->
place_in_slider
```

F3 answers:

```text
What is the current active subgoal?
Is it complete?
What is the next subgoal?
```

The first experiments should use an **oracle F3** based on valid ground-truth annotation boundaries or task-success signals.

This isolates the main scientific question:

> **If the switch time is known perfectly, can latent path planning connect the current action region to the next one?**

Only after this succeeds should a learned completion model be introduced.

A learned F3 can later predict:

```text
p(subgoal_complete)
```

using only causal, repository-supported inputs.

---

# 8. F3 as execution memory

F3 should also maintain a waypoint execution trace:

```text
T = {W_0, W_1, ..., W_K}
```

Every waypoint must use exact robot/control fields available in the repository.

At every subgoal transition, save a branch checkpoint:

```text
W_branch
```

The first `return` command should mean:

> **Return the robot to the most recent recorded branch checkpoint by replaying or tracking the stored waypoint trace in reverse.**

The initial claim must be conservative:

```text
robot execution-state return
```

not:

```text
exact world-state reversal
```

Possible later variants include:

- reverse raw action replay;
- reverse joint-waypoint replay;
- reverse Cartesian waypoint tracking;
- MPC-guided reverse waypoint tracking.

Object-state recovery should be measured separately.

---

# 9. Normal execution logic

The intended stack is:

```text
long-horizon language
       ↓
      F3
active atomic subgoal
       ↓
      F1
nominal local latent continuation
       ↓
      F2
feasible / smooth latent trajectory
       ↓
frozen action decoder
       ↓
robot action
```

Inside an atomic action:

```text
F3 keeps the current subgoal
F1 predicts local nominal continuation
F2 keeps the executed latent/action trajectory feasible and smooth
```

At a subgoal boundary:

```text
F3 switches the target action region
F2 temporarily prioritizes the new target/reference
F2 plans a continuous transition toward the new action region
F1 remains a local dynamics prior, not a task-switching model
```

After the trajectory enters the new action region:

```text
F1 again provides local nominal evolution
F2 continues receding-horizon control
```

---

# 10. Why this differs from Wave28–Wave78

The old question was:

```text
What displacement should modify the current latent so that it becomes more like the next action?
```

The new question is:

```text
What sequence of locally feasible latent states connects the current executable action coordinate to a target action region?
```

Old formulation:

```text
z' = z + delta_z
```

New formulation:

```text
trajectory* =
argmin over {z_1 ... z_H}
    terminal cost
  + dynamics cost
  + continuity cost
  + executability/support cost
```

Pointwise steering can fail even when path planning is possible.

Wave28–Wave78 should therefore remain in the paper as a negative intervention study and as motivation for the new formulation.

---

# 11. First new scientific test

The new series should begin with **oracle endpoint / oracle switch latent path completion**.

Use real continuous trajectories containing ordered transitions such as:

```text
lift -> place
```

Use the actual repository task identifiers discovered from data.

Construct:

```text
start latent = real latent near the end of Action A
goal set     = valid Action-B latent region
```

Hide the real intermediate path from the planner.

Provide only:

```text
start
goal region
frozen representation
frozen decoder
frozen F1 dynamics prior
```

Compare:

- linear interpolation;
- F1 free rollout;
- old F2 local refinement;
- best Wave28–Wave78 pointwise steering baseline;
- graph / kNN path planning;
- trajectory optimization without F1;
- trajectory optimization without continuity;
- trajectory optimization without support/executability;
- new F2-MPC-style planning;
- hybrid graph + continuous refinement if feasible.

The first question is:

> **Does the learned action coordinate space support constrained path completion between atomic action regions?**

If this fails even with oracle start/goal/switch information, there is no reason to add a learned F3 yet.

---

# 12. Second stage: target-region planning

If oracle endpoint path completion succeeds, remove exact endpoint knowledge.

Give:

```text
current latent
Action-B-compatible terminal set
```

and let the planner choose the endpoint.

This tests whether the space supports navigation toward an action region instead of one memorized future latent.

---

# 13. Third stage: language-conditioned terminal planning

If target-region planning succeeds, derive the target from the frozen language-action representation.

The planner receives language for Action B rather than the exact terminal latent.

This directly connects the representation result to control.

---

# 14. Fourth stage: closed-loop hierarchical execution

Only after the above stages succeed should the system perform full closed-loop execution:

```text
execute current atomic action
detect completion
switch target
plan H-step latent trajectory
execute a short prefix
observe/update
replan
repeat
```

At this point F2 becomes legitimately MPC-like because it performs:

```text
plan -> execute prefix -> observe -> replan
```

---

# 15. Return stage

Once forward hierarchical execution works:

```text
branch -> execute new subgoal -> return
```

F3 records the branch trace and replays/tracks it backward on return.

Evaluate:

```text
final joint error
final end-effector error
maximum tracking deviation
action smoothness / jerk
collision count
gripper-state recovery
object-state error separately
```

Do not describe waypoint replay as physical time reversal.

---

# 16. Revised paper story

The revised story is:

> **Actions can be represented as language-addressable, continuously decodable coordinates. Local dynamics predict how these coordinates naturally evolve, and iterative refinement stabilizes local rollout. However, extensive post-hoc steering experiments show that language-addressable coordinates are not automatically pointwise editable. This motivates a control formulation: high-level language selects atomic action regions, local dynamics define feasible motion, and a trajectory optimizer plans continuous paths between action coordinates. A task executive switches subgoals and records execution traces for lightweight return.**

Compact framing:

> **Actions are coordinates. Dynamics tell us how coordinates move. Planning tells us how to travel between them.**

Alternative framing:

> **From action representation to action-space navigation.**

---

# 17. Revised contribution structure

**Representation contribution**

A continuous action latent jointly supports language addressability and motor fidelity.

**Dynamics contribution**

A learned local latent dynamics model predicts coordinate evolution, and iterative correction stabilizes rollout.

**Negative intervention contribution**

Broad pointwise steering fails to provide reliable continuous action-to-action retargeting, showing that addressability does not imply local editability.

**Planning contribution**

A latent trajectory controller connects atomic action regions by optimizing multi-step paths under learned dynamics, continuity, and empirical execution-support constraints.

**Hierarchy contribution**

A task executive sequences long-horizon subgoals, switches targets after subgoal completion, and maintains execution memory.

**Return contribution**

Recorded execution traces provide lightweight memory-conditioned return to previous robot execution states.

---

# 18. What should remain frozen initially

For the first new experiment series:

```text
freeze the action representation
freeze its decoder
freeze language projection / semantic structure
freeze validated representation checkpoints
freeze the current F1 initially
retain the current F2 as a historical baseline
```

Do not simultaneously retrain everything.

The first new optimization/training target should be the new trajectory-planning F2 or an explicitly defined path optimizer.

F3 should initially be oracle.

---

# 19. What may be retrained later

Only after a specific bottleneck is identified should later experiments consider:

```text
controlled F1 dynamics
goal-conditioned F1
multi-modal F1 transition distributions
state-conditioned F1
learned F3 completion
learned target-set selection
new path-support critics
state-conditioned hybrid latent planning
```

Each change must answer a distinct failure mechanism.

Do not reopen the representation unless a new experiment directly shows that the frozen latent itself lacks the connectivity required for planning.

---

# 20. Scientific success criteria

A defensible final result should eventually demonstrate:

```text
A. latent path planning beats interpolation and pointwise steering;
B. planned paths remain close to empirical executable support;
C. decoded action continuity remains acceptable;
D. endpoints enter the correct next-action region;
E. the planner works across multiple ordered atomic transitions;
F. closed-loop replanning improves over open-loop latent rollout;
G. F3 can sequence subgoals without manual atomic commands;
H. long-horizon composed tasks succeed;
I. waypoint-trace return restores the robot to recorded execution states;
J. ablations support distinct roles for F1, F2, and F3.
```

---

# 21. Main failure interpretations

If oracle path planning fails even with a known start and known target latent/region:

> the frozen action representation may not contain sufficiently connected executable transition geometry.

If oracle path planning succeeds but language-conditioned target selection fails:

> semantic target grounding is the bottleneck.

If offline paths look good but decoded execution is discontinuous:

> the executability / control interface or planning objective is insufficient.

If open-loop planning succeeds but closed-loop execution fails:

> state feedback and model mismatch are the bottleneck.

If learned F3 fails while oracle F3 succeeds:

> progress / completion estimation is the bottleneck.

If reverse replay returns robot configuration but not object state:

> the result supports execution-state return, not full environment reversal.

---

# 22. Final strategic position

Do not continue trying to prove:

> a low-dimensional residual field can directly retarget a frozen action latent.

The project should now attempt to prove:

> **a language-grounded action representation can serve as a navigable control coordinate system when local dynamics, trajectory planning, and task-level navigation are separated.**

The long-term architecture is:

```text
Frozen Action Coordinates
        +
F1 Local Dynamics
        +
F2 Latent MPC / Path Planner
        +
F3 Task Executive + Waypoint Memory
```

This is the new post-Wave78 research direction.
