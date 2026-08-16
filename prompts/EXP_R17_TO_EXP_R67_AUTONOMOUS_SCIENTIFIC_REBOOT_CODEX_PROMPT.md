# EXP_R17–EXP_R67 Autonomous Codex Research Program
## Scientific Reboot After the R17–R58 Gate-Only Failure
## Hierarchical Latent Control with F1 Local Dynamics, F2 Latent MPC, F3 Task Executive, and Waypoint Return

# 0. Program reset

You are Codex operating inside the existing **Actions as Coordinates** research repository.

The prior program produced meaningful scientific experiments through:

```text
EXP_R16
```

However, the previous records labeled:

```text
EXP_R17 ... EXP_R58
```

were largely stage-specific interface gates, data-feasibility checks, and repeated confirmations that exact causal simulator/controller state was unavailable.

Those gate-only runs do **not** count as valid scientific experiments for the new program.

This document resets the scientific experiment sequence at:

```text
EXP_R17
```

The new autonomous program may continue through at most:

```text
EXP_R67
```

The hard stop is:

```text
EXP_R67
```

Do not start:

```text
EXP_R68
```

Stop earlier if the full target system is scientifically achieved.

# 1. Remove the old gate-only R17–R58 experiment history

Before starting the new EXP_R17, inspect the repository and identify all artifacts that belong exclusively to the previous gate-only EXP_R17–EXP_R58 stage.

Delete or retire from the active experiment history only those artifacts that exist solely because the old program consumed EXP_R17–EXP_R58 without introducing a genuinely new scientific experiment.

Examples may include old `reports/EXP_R17_report.md` through `reports/EXP_R58_report.md`, old `next_exp_fromR17.md` through `next_exp_fromR58.md`, gate-only preregistrations, repeated interface-check summaries, and final summaries that imply 50 complete scientific experiments were executed when most were only gates.

Do not delete EXP_R1–EXP_R16 scientific results, EXP_R8 success artifacts, representation checkpoints, F1 checkpoints, historical F2 checkpoints, datasets, source code, tests, data manifests, or real diagnostic results from R9–R16.

If a file mixes useful R9–R16 evidence with gate-only R17–R58 bookkeeping, preserve the useful content and remove only the invalid gate-only experiment interpretation.

Preserve Git history. Do not rewrite repository history.

The goal is:

> **R17 must again become the next genuine scientific experiment after R16.**

# 2. What counts as a real experiment

A new EXP_R experiment counts only if it introduces at least one genuinely new scientific element: a new scientific hypothesis, model family, control formulation, planning formulation, loss/cost structure, dataset construction, causal surrogate, evaluation protocol, mechanism-level ablation, F1/F2/F3 interaction design, return/control mechanism, or prospective data-collection protocol enabling a new test.

The following do not count as separate scientific experiments: re-running the same unavailable-interface check, reconfirming the same missing simulator/controller state, repeating the same data-stop diagnosis, renaming the same failed experiment, changing only one scalar without a mechanism-level reason, changing only learning rate/batch size, bookkeeping-only runs, or documentation-only runs.

A repeated data/interface gate may consume **at most one experiment ID**. After that, Codex must pivot to a valid surrogate problem, redesign the model, redesign the benchmark, redesign the data construction, search for an alternative method, or define a new prospective collection protocol.

# 3. Preserved scientific state

Representation remains frozen initially and provides language-addressable, motor-faithful action coordinates.

F1 remains a learned local latent dynamics predictor/prior. It predicts where the current atomic action trajectory naturally evolves next and does not perform task switching.

Historical F2 remains a local iterative refiner and frozen baseline. It is not yet MPC.

EXP_R8 established a supported offline multi-step path-planning result. The selected method was `repair_late_0.75`, with held-out arrival rate 1.0000, decoded first difference 1.0375, and hidden path MSE 0.9743.

Preserve the interpretation:

> **Action transitions are paths, not displacements.**

EXP_R9–EXP_R16 remain valid surrogate-control diagnostics. Do not delete them.

# 4. Final target system

The R17–R67 program must work toward four capabilities.

## Goal 1 — Stable atomic action execution

When the current subgoal is unfinished, the robot should continue the current atomic action stably. Future subgoals must not prematurely corrupt the unfinished action. F1 provides local progression and F2 maintains smooth executable latent/action evolution.

## Goal 2 — Action A to Action B through a latent path

Transitions must occur through a multi-step latent path:

```text
z_A -> z_1 -> z_2 -> ... -> z_H -> G_B
```

where `z_A` is the current executable latent and `G_B` is a target Action-B-compatible region. Do not return to pointwise force-field steering.

## Goal 3 — F2 becomes true latent MPC

A legitimate F2-MPC loop must perform:

```text
observe
encode current state
plan H steps
execute only a short prefix
observe resulting state
re-encode
replan
repeat
```

Do not call an open-loop path generator MPC.

## Goal 4 — F3 controls long-horizon task progression

F3 becomes the high-level Task Executive. Given a long-horizon instruction such as `lift the blue block, then place it in the slider`, F3 must maintain the active subgoal, detect completion, switch to the next subgoal only after completion, pass the new target region to F2, manage branch checkpoints, and maintain waypoint execution memory.

# 5. F3 waypoint memory and return

F3 maintains an execution trace `T = {W_0, W_1, ..., W_K}` using exact repository-supported robot/controller fields. At subgoal transitions save a `branch_checkpoint`.

For the first `return` implementation:

```text
stop forward progression
select the relevant checkpoint
reverse the waypoint reference
track/replay the recorded trajectory backward
```

The first claim is only:

> **return to a previously recorded robot execution state**

Do not claim exact world-state reversal unless directly measured.

# 6. Key correction to the previous autonomous program

The previous program treated the lack of exact causal simulator/controller state as a reason to stop scientific exploration.

The new program must not do that.

If exact physical MPC cannot be certified from existing data, continue with the strongest valid next scientific question. Possible directions include offline latent MPC structure, controlled F1, goal-conditioned F1, multi-hypothesis F1, uncertainty-aware planning, latent transition graphs, multiple shooting, terminal-set MPC, adaptive horizon, phase-dependent authority, value terminal cost, robust/tube-style MPC surrogates, learned support critics, F3 progress modeling, hazard-based completion, change-point detection, sequence-model completion, joint F2/F3 timing analysis, counterfactual offline benchmarks, waypoint-return surrogates, and prospective data collection.

A missing simulator snapshot must never consume dozens of experiment IDs.

# 7. New EXP_R17 starting question

EXP_R17 must be a genuine scientific experiment, not an interface audit.

Recommended question:

> **Why did EXP_R8 late repair succeed, and can that fixed repair rule be converted into a principled horizon-dependent control law that generalizes across action pairs?**

R17 should test whether the empirical pattern:

```text
preserve local trajectory early
increase target authority late
```

is a general control principle.

# 8. EXP_R17 broad method tournament

At minimum, where valid, compare:

```text
fixed R8 repair_late_0.75
linear horizon-dependent goal weighting
piecewise early/late weighting
sigmoid goal-authority schedule
distance-to-target adaptive weighting
F1-confidence adaptive weighting
proposal-confidence adaptive weighting
uncertainty-aware weighting
learned horizon-position weighting
task-pair-conditioned weighting
small controller predicting per-step repair authority
```

Do not reduce R17 to tuning one coefficient.

Determine whether late repair is universal, task-pair dependent, distance dependent, confidence dependent, or phase dependent.

# 9. F2 planning objective family

Use a structured cost of the form:

```text
J =
    lambda_goal(k) * J_goal
  + lambda_dyn(k) * J_dyn
  + lambda_cont * J_cont
  + lambda_exec * J_exec
  + lambda_support * J_support
  + lambda_path * J_path
  + lambda_curve * J_curve
```

The central variable is whether `lambda_goal(k)` and `lambda_dyn(k)` should depend on horizon position or state.

# 10. Candidate future F2 directions

If R17 does not solve the problem, later experiments should explore genuinely different families such as gradient trajectory optimization, multiple shooting, constrained optimization, augmented Lagrangian methods, trust-region planning, CEM, MPPI, random shooting, population-based search, kNN/A*/Dijkstra latent graphs, graph+local MPC, uncertainty-aware MPC, ensemble F1, tube-style latent MPC, distributional terminal cost, trajectory proposal networks, goal-conditioned proposal models, planner distillation, proposal+optimization, global graph+local MPC, retrieval+repair, and multi-resolution planning.

Each EXP should compare several methods when feasible.

# 11. Candidate future F1 directions

Keep F1 frozen until evidence identifies it as a bottleneck.

If needed, test genuinely different models such as goal-conditioned F1, controlled F1, multi-hypothesis F1, mixture-density F1, ensemble F1, history-conditioned F1, state-conditioned F1, phase-conditioned F1, task-pair-conditioned F1, or a transition-distribution model.

Always compare against the frozen historical F1.

# 12. Candidate future F3 directions

Do not abandon F3 because one readiness diagnostic failed.

Possible future experiments include binary completion classification, progress regression, temporal MLP, GRU, Transformer history models, hazard models, change-point detectors, boundary likelihood models, semantic-progress models, execution-progress models, F1-error-based completion, F2-target-distance-based completion, and multi-signal fusion.

Evaluate AUROC, AUPRC, balanced accuracy, early-switch rate, late-switch rate, switch delay, false completion, missed completion, task-pair generalization, source-session generalization, and downstream execution effects.

# 13. Long-horizon task experiments

Once F2 is reliable under oracle F3, integrate learned F3.

Start with two-step tasks, then extend only after two-step execution is stable.

The system should receive one long-horizon instruction, maintain the active subgoal, execute the current atomic action, detect completion, switch goal, plan the latent transition, and continue without human intermediate commands.

# 14. Return experiments

Return is a separate layer and must not block earlier progress.

Test, where valid:

```text
reverse action replay
reverse joint-waypoint replay
Cartesian waypoint tracking
MPC-guided waypoint return
```

Measure joint error, end-effector error, gripper-state error, tracking deviation, jerk, collision/safety, return success, and object-state error separately.

If exact controller snapshots are unavailable, define a prospective data-collection protocol or the strongest valid surrogate. Do not spend multiple EXP IDs repeating the same limitation.

# 15. Prospective data collection is allowed

If existing data fundamentally cannot answer a control question, Codex may define a new prospective collection experiment.

Such an EXP must state exactly what causal variable is missing, why it blocks the hypothesis, what fields must be collected, what simulator/robot state must be saved, what action prefix must be logged, what waypoint/checkpoint fields are needed, what branch/counterfactual protocol should be used, how many trajectories are required, what train/dev/held-out split will be used, and what success gate follows collection.

A data-collection experiment is valid only if it enables a new scientific test.

# 16. Required preregistration

Before every experiment create:

```text
reports/EXP_R{id}_preregistration.md
```

It must state the hypothesis, methods, frozen components, trainable components, data, splits, metrics, success gate, failure interpretation, and held-out opening rule.

Do not change the hypothesis after held-out evaluation.

# 17. Required report

After every experiment create:

```text
reports/EXP_R{id}_report.md
```

For the reboot:

```text
reports/EXP_R17_report.md
```

Each report must include the scientific question, motivation, previous bottleneck, repository commit, environment, data provenance, frozen hashes, exact interfaces, methods, architectures, control/planning formulations, losses/costs, hyperparameter ranges, runs, failures, development results, held-out results, ablations, statistics, qualitative failures, figures, tables, claim decision, and remaining bottleneck.

Negative results must remain.

# 18. Required next-experiment document

After every experiment create:

```text
reports/next_exp_fromR{id}.md
```

For R17:

```text
reports/next_exp_fromR17.md
```

It must explain what was established, what failed, what remains unresolved, the most likely mechanism, what the next experiment should test, why it is justified, what remains frozen, what may change, mandatory baselines, new methods, success criteria, and falsification criteria.

# 19. External research

Codex is authorized and encouraged to search the internet when methods fail or implementation details are unclear.

Prefer primary sources: conference papers, arXiv, official project pages, official GitHub repositories, and official documentation.

Relevant topics include latent MPC, TD-MPC/TD-MPC2, CEM, MPPI, iLQR, multiple shooting, hierarchical MPC, robust/tube MPC, model predictive path integral control, goal-conditioned control, latent world-model planning, hierarchical RL, options, skill termination, task-progress prediction, change-point detection, waypoint tracking, and replay control.

Whenever external research is used, create:

```text
reports/EXP_R{id}_external_research.md
```

recording query, paper/repository title, URL, access date, year, commit/tag if relevant, borrowed idea, why it is relevant, and how it was adapted.

# 20. Broad-experiment rule

Each EXP should explore multiple meaningful directions when computationally feasible.

Do not create a sequence where each experiment changes only one learning rate, one lambda, or one batch size unless debugging a concrete numerical failure.

Prefer one EXP containing multiple model families, planners, objectives, horizon structures, or ablations testing the same central hypothesis.

# 21. Experiment-budget integrity

An EXP ID may be consumed only if a real method was implemented/evaluated, a real scientific benchmark/data construction was completed, or a genuinely new causal hypothesis was tested.

The same missing snapshot/contact/controller state/teacher-forcing limitation cannot consume multiple IDs.

One gate is enough. After that, pivot.

# 22. Held-out discipline

Every EXP must separate train/development from held-out/final evaluation.

Before held-out, freeze the candidate, metrics, thresholds, seeds, manifests, and checkpoint hashes.

Open held-out once.

If held-out fails, preserve the result, write the report, write next_exp, and continue to the next genuine experiment.

Do not repair the same experiment after seeing held-out.

# 23. Autonomous loop

After every experiment:

```text
1. finish development
2. freeze candidates
3. open held-out once
4. write EXP report
5. write next-exp document
6. diagnose the mechanism
7. search literature if useful
8. define the next genuine experiment
9. preregister
10. run
```

Continue autonomously from R17 through R67.

Do not ask for user confirmation between experiments.

# 24. Success condition

Do not stop because one offline metric improves.

Overall `SUCCESS` requires a defensible version of the target system, including most of:

```text
atomic actions remain stable before subgoal completion
Action A -> Action B uses multi-step latent paths
F2 performs real receding-horizon replanning
closed-loop F2 preserves/improves target arrival and continuity
multiple atomic transition pairs work
F3 detects subgoal completion
F3 automatically switches subgoals
one long-horizon instruction executes multiple atomic actions without manual intermediate commands
future goals do not prematurely corrupt unfinished current actions
F3 records waypoint execution memory
return restores robot execution state to a recorded checkpoint under a valid controller/simulator setup
ablations support distinct F1/F2/F3 roles
held-out/prospective evidence supports the claims
```

Stage-level success labels such as `F2_MPC_SUPPORTED`, `F3_COMPLETION_SUPPORTED`, `LONG_HORIZON_TWO_STEP_SUPPORTED`, and `RETURN_SUPPORTED` may be used, but do not stop until overall success or R67.

# 25. Final target demonstration

The mature system should support a command such as:

```text
lift the blue block, then place it in the slider
```

with:

```text
F3 selects lift
F1 predicts local lift progression
F2 performs smooth executable local control
F3 detects lift completion
F3 switches target to place
F2 plans a latent transition path
F2 executes a short prefix
system observes actual state
system re-encodes
F2 replans
repeat until place completes
```

For `return`:

```text
F3 selects the stored checkpoint
F3 loads the waypoint trace
trace is reversed
controller/F2 tracks the reference
robot returns to the recorded execution state
```

# 26. Paper-level target

The mature paper should support:

> **Actions are language-addressable coordinates. F1 models local motion, F2 plans and controls multi-step latent paths, and F3 navigates long-horizon subgoals while maintaining execution memory.**

Core transition claim:

> **Action transitions are paths, not displacements.**

Core control claim:

> **A language-grounded action representation becomes a navigable control space when local dynamics, trajectory control, and task-level navigation are separated.**

# 27. Failure must lead to action

Every failure must be converted into one of:

```text
new method
new ablation
new surrogate
new benchmark
new model family
new data construction
new prospective collection protocol
```

Do not stop merely because one dataset cannot prove one physical claim.

Stop only when overall SUCCESS is achieved or EXP_R67 is completed.

# 28. Hard stop

Continue:

```text
EXP_R17
EXP_R18
...
EXP_R67
```

If success occurs earlier, stop and generate final artifacts.

If EXP_R67 completes without full success:

```text
do not start EXP_R68
```

Generate:

```text
FINAL_R17_R67_RESEARCH_SUMMARY.md
FINAL_R17_R67_FAILURE_TAXONOMY.md
FINAL_R17_R67_SUPPORTED_CLAIMS.md
FINAL_R17_R67_BEST_SYSTEM.md
FINAL_R17_R67_RECOMMENDED_NEXT_DIRECTION.md
```

Do not force a positive conclusion.

# 29. Required final artifacts on success

If the target system succeeds before R67, generate:

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

Then stop.

# 30. Start now

Begin by cleaning the old gate-only R17–R58 experiment history from the active experiment sequence.

Then create:

```text
reports/EXP_R17_preregistration.md
```

and begin the new scientific EXP_R17.

The first new milestone is:

> **Determine whether the successful R8 late-repair behavior reflects a general horizon-dependent control principle, and use that result to build a stronger F2 trajectory controller rather than continuing fixed repair heuristics.**

Begin EXP_R17 now.
