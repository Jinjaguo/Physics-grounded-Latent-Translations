# ACTIONS AS COORDINATES: FINAL PAPER METHOD OUTLINE AND END-TO-END RESEARCH GOAL

## Paper Position

This project studies whether learned action representations can become a controllable execution space for embodied agents. The central premise is that a robot should not need to predict a complete future world trajectory before every manipulation decision. Instead, language should identify the intended action semantics, the learned action latent should represent where the robot currently is inside the execution of that action, and a closed-loop controller should use physically realized feedback to advance, refine, switch, and recover.

The paper therefore treats action latents as **coordinates for control**. A coordinate is useful only if it supports intervention. The system must be able to receive an ordered instruction such as `lift_blue_block_slider -> place_in_slider`, continue executing `lift_blue_block_slider` without being destabilized by the future `place_in_slider` goal, detect when the current atomic action is actually complete, redirect control toward `place_in_slider` from the robot's physically reached current state, and continue the next action without restarting the trajectory from the original initial condition. The final paper must establish this behavior with closed-loop action-conditioned feedback rather than teacher-forced latent rollouts.

## Core Scientific Question

The final scientific question is:

> Can a learned action latent space serve as a programmable closed-loop control interface in which language selects action semantics, local latent dynamics advance the currently active action, refinement keeps execution physically realizable, and a switching mechanism composes multiple atomic actions from the robot's actual current state?

The paper should answer this question at the system level and at the mechanism level. System-level evidence must show that the full pipeline can execute ordered multi-action instructions under closed-loop feedback. Mechanism-level evidence must show why the behavior works: which representation properties are required, what F1 contributes, what F2 contributes, what F3 contributes, how current-action protection is maintained, how target switching is triggered, how closed-loop feedback corrects drift, and which failure modes remain.

## Final System Abstraction

Let the robot observation at time \(t\) be \(o_t\), the robot proprioceptive or configuration state be \(q_t\), the active language action be \(g_t\), and the learned action representation be \(z_t = E(o_t, q_t, g_t)\), where \(E\) is the action-state encoder. The latent state \(z_t\) is therefore an execution representation conditioned on what the robot is doing and on the physically observed state.

The robot receives an ordered action program \(G = [g^{(1)}, g^{(2)}, \ldots, g^{(K)}]\). The controller maintains an active action index \(k_t\). During execution of \(g^{(k_t)}\), future actions may be known to the executive layer, but they must not prematurely dominate low-level control. The system must preserve the active action until evidence indicates that the current action has completed or that a safety/recovery condition requires intervention.

The system contains three functional modules. **F1 is the local action-progress model.** Its role is to predict a locally appropriate latent or action update for the currently active atomic action. F1 should preserve short-horizon execution stability and should remain conditioned primarily on the current action. **F2 is the closed-loop refinement and local planning module.** Its role is to evaluate or optimize short latent/action proposals, execute an actual control intervention through the simulator or robot controller, observe the resulting next physical state, re-encode that state, and refine again. F2 must therefore operate through the causal loop `proposal -> executed action -> physical/simulated next state -> observation -> re-encoding -> replanning`. **F3 is the action-completion and switching executive.** Its role is to determine when the current action has finished, increment the action program from \(g^{(k)}\) to \(g^{(k+1)}\), and give the next action authority only after completion evidence is sufficient. F3 may initially be evaluated with an oracle boundary to isolate F2, but the final system must replace oracle switching with a learned or otherwise autonomous completion mechanism.

The architecture may also maintain **checkpoint and waypoint memory**. This memory stores physically meaningful execution states, controller state, robot configuration, observations, and corresponding latent states at selected moments. Its purpose is recovery and branch management. Any claim of return or recovery must be evaluated by actually executing control back toward a stored physical state; latent-space reversal alone is insufficient evidence.

## Closed-Loop Execution Pipeline

At the beginning of an episode, the instruction sequence is parsed into ordered atomic actions. The system encodes the current observation, robot state, and active action into \(z_t\). F1 proposes a local progress update for the current action. F2 generates one or more refined executable control proposals and chooses among them using explicit objectives such as current-action progress, target-action readiness when appropriate, execution continuity, support under the data distribution, constraint satisfaction, uncertainty, and short-horizon outcome quality.

The chosen action is then sent to the actual simulator/controller interface. The environment returns the physically realized next observation and robot state. The system re-encodes these observations to obtain the next latent state. The next planning step must use this realized state rather than a teacher-forced latent taken from an offline trajectory. This feedback loop continues until F3 determines that the active atomic action is complete.

Only then does the system update the active language action to the next element of the ordered program. Retargeting must begin from the robot's current physical state and current re-encoded latent. The controller must not regenerate the second action from the initial state of the episode. The same closed-loop process then continues for the next action. In the final integrated demonstration, an instruction such as `lift -> place` should therefore appear as one continuous execution with a stable lift phase, an evidence-driven switching event, and a closed-loop place phase.

## Current-Action Protection

A central mechanism in the final paper is **current-action protection**. When the robot is executing an atomic action, the presence of future actions must not cause premature target attraction or latent drift. For example, when `lift` is active and `place` is the next action, the controller must continue to satisfy the execution requirements of `lift` until completion. Future-target information may be used by the executive layer for planning readiness, but it must be gated so that it cannot corrupt the current action.

This mechanism should be measured directly. The experiment should compare current-action trajectories with and without future-action conditioning and quantify action completion, deviation from current-action execution, premature switching frequency, continuity, task success, and downstream composition success. A successful mechanism should preserve the current action while still allowing rapid transition after completion.

## F1: Local Action Progress

F1 should answer the local question: given the robot's current execution representation and the currently active action, what local progress is plausible next? Its value lies in short-horizon stability, local continuity, and preservation of the active action. The project already has evidence that local prediction/refinement structure is useful, but the final paper must test F1 inside the causal closed-loop execution stack.

The final claim for F1 should be supported only if removing or replacing F1 measurably harms local execution stability, continuity, current-action completion, or the quality of the states presented to F2. If F2 alone consistently dominates F1, the final architecture should be simplified rather than preserving F1 for narrative symmetry.

## F2: Closed-Loop Refinement and Local Planning

F2 is the most important missing causal component. The final version must operate on action-conditioned environment feedback. A proposal is meaningful only after the corresponding action is executed and the resulting state is observed.

F2 may use retrieval, local optimization, graph planning, MPC-like short-horizon optimization, stochastic sampling, learned transition models, ensembles, value functions, support critics, uncertainty-aware objectives, or combinations of these methods. The specific formulation should be selected empirically. The invariant requirement is causal execution: every claimed refinement improvement must ultimately be evaluated through actual controller steps.

The primary F2 experiment should first use oracle F3 boundaries so that switching errors do not obscure whether closed-loop latent refinement itself works. It should compare strong local baselines, open-loop or teacher-forced variants, and the best closed-loop formulation under matched initial states. The desired evidence is that F2 improves realized action completion, continuity, robustness to perturbation, and transition quality after the environment has responded to the proposed controls.

## F3: Completion Detection and Ordered Switching

F3 answers when authority should move from the current atomic action to the next one. This is a control decision, not merely a classification score. Premature switching can corrupt the current action, while delayed switching can waste steps or move the robot away from a useful transition state.

F3 may use latent history, robot state, progress features, task-specific completion observations, learned temporal models, hazard-style completion models, calibrated confidence, or hybrid semantic-execution signals. The final evaluation must test the switching mechanism inside closed-loop two-step and longer ordered tasks.

The final claim should require low premature-switch and late-switch rates together with successful downstream execution. Accuracy on offline boundary labels alone is insufficient. The system must demonstrate that learned F3 can replace oracle F3 with a bounded degradation in end-to-end success.

## State Memory, Recovery, and Return

Recovery should be treated as an execution problem. At selected points the system may store a checkpoint containing physical robot state, controller state, observations, latent state, active action, action-program index, and other simulator state required for reproducibility. When recovery is requested, the robot must execute a return controller and be evaluated after physical/simulated rollout.

The strongest final result would show branch recovery or return to a previously useful checkpoint followed by successful continuation. If this cannot be made sufficiently reliable within the paper scope, return should remain a secondary analysis rather than a central claim. The paper must never equate geometric proximity in latent space with physical state recovery.

## Required Data and Infrastructure

The decisive dataset must contain **action-conditioned transitions**. Each transition should link a complete current state snapshot, the proposed or executed action, the next simulator or robot state, the next observation, the active language action, completion status when available, and all state required to branch or replay the controller. Matched-current-state interventions are especially valuable because different control proposals can then be compared from the same causal starting point.

The simulator/controller interface must therefore support state reset or checkpoint restoration, action execution, observation capture, robot-state capture, deterministic or controlled stochastic replay when possible, and logging of every intervention. If the current environment lacks one of these functions, infrastructure work is part of the experiment. It must be implemented before the corresponding scientific question is declared untestable.

## Experimental Structure

The experimental program should progress from causal local control to integrated composition. First, establish an action-conditioned closed-loop benchmark in which different F2 proposals can be executed from matched states. Then determine which F2 family actually improves realized next-state and short-horizon control. Next, test two-step composition with oracle F3 so that the effect of current-action protection and retargeting can be isolated. After F2 and two-step composition are stable, train and evaluate learned F3. Then test three-step or longer ordered composition, perturbation robustness, and recovery/checkpoint mechanisms. Finally, run integrated ablations that remove or replace F1, F2, F3, current-action protection, causal feedback, and memory.

This ordering is methodological rather than bureaucratic. If an intermediate mechanism fails, the experimental program must change the mechanism and continue attacking the same scientific requirement. It must not accumulate experiment numbers for unexecuted ideas.

## Baselines and Ablations

The final paper should compare the integrated method against meaningful alternatives. At minimum, the evidence should distinguish closed-loop causal feedback from teacher-forced latent progression, current-state retargeting from restart-from-initial-state execution, active-action protection from unrestricted future-goal conditioning, oracle F3 from learned F3, and the integrated stack from its principal module removals.

Additional planner families should be included when they test genuine mechanism differences. Repeated coefficient sweeps around the same formulation should be treated as tuning inside one experiment rather than separate experiments. Multiple implementations, model families, objectives, and control formulations may be evaluated within the same EXP when they address the same scientific question.

## Primary Evaluation

The final system should report atomic-action success, ordered multi-action success, current-action preservation, premature and delayed switching, physical-state continuity, recovery after perturbation, intervention efficiency, closed-loop stability, and computational cost. Where stochastic evaluation is involved, results should include repeated trials, seeds or repeated matched-state interventions, uncertainty intervals where appropriate, and per-task breakdowns.

Every reported gain must be traceable to machine-verifiable artifacts such as raw rollout logs, intervention records, model checkpoints, evaluation files, plots generated from saved metrics, exact run commands or configurations, code revision identifiers, and environment information.

## Final Acceptance Goal

The project is complete only when the paper can make the following end-to-end claim with direct experimental evidence:

> Given an ordered language instruction composed of multiple atomic manipulation actions, the system can preserve and complete the currently active action, use action-conditioned closed-loop feedback to refine execution from the robot's physically realized state, autonomously detect completion, switch to the next language-specified action without restarting from the original state, and execute the ordered sequence with measurable advantages over open-loop, teacher-forced, restart-based, and module-ablated baselines.

The minimum convincing integrated demonstration is a **closed-loop two-action sequence such as `lift -> place` with autonomous switching**, evaluated across enough held-out initial conditions and perturbations to establish repeatability. The stronger target is multi-action composition with checkpoint-based recovery.

The paper must also establish that the latent representation is doing control-relevant work. A successful system that can be reproduced without using the latent representation would not support the intended scientific contribution. Mechanism ablations must therefore show that intervention in the learned action representation contributes to execution, refinement, switching, or recovery beyond conventional baselines.

## Claim Discipline

Offline latent interpolation, teacher-forced next latents, surrogate trajectory distance, synthetic plants, or latent waypoint reversal may be used for development and diagnostics, but they cannot by themselves support claims of robot closed-loop control. A method is considered physically or causally validated only when its action is executed through the simulator or robot controller and its next decision uses the resulting observed state.

The final paper should prefer a smaller number of deeply validated claims over a larger number of loosely supported mechanisms. Any module that does not survive closed-loop ablation should be removed or reframed. The endpoint is an experimentally defensible control system and a clear scientific result about when and how an action latent can function as a programmable coordinate system for embodied execution.
