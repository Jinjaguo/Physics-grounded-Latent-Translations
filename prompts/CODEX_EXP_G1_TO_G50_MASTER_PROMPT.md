# CODEX MASTER PROMPT: EXP_G1–EXP_G50 CLOSED-LOOP SCIENTIFIC EXECUTION PROGRAM

You are restarting the experimental program from scratch under a new naming scheme. Do not continue any previous `EXP_R*`, `Wave*`, or other experiment numbering. The first valid experiment is `EXP_G1`. The `G` series stands for the new generalizing, causally executed experimental program.

Your job is to autonomously execute a sequence of real experiments that advances the research objective defined in `ACTIONS_AS_COORDINATES_FINAL_METHOD_AND_GOAL.md`. You must continue iterating until the final acceptance goal in that document has been achieved with machine-verifiable evidence, or until **50 valid executed experiments** have been completed. The hard upper bound is therefore `EXP_G50`.

The governing research objective is to build and validate a closed-loop action-latent control system in which language specifies an ordered sequence of atomic actions, the current action is preserved until completion, F1 advances local execution, F2 performs causal closed-loop refinement using physically realized simulator/controller feedback, F3 detects completion and switches to the next action, retargeting begins from the current physical state, and the integrated system can execute at least a robust `lift -> place` sequence with autonomous switching. Stronger long-horizon composition and checkpoint-based recovery should follow when the core system is stable.

## Non-Negotiable Definition of a Valid EXP ID

An `EXP_G{id}` number may be consumed only when a real scientific experiment has actually executed and produced new evidence.

A valid EXP must do at least one of the following in an actually executed way: generate new data; train or fit a model; execute a new intervention; execute a simulator/controller rollout under a new control formulation; construct and evaluate a new dataset; run a new evaluation protocol on newly produced or newly transformed evidence; or perform a mechanism-level ablation that changes the causal system being tested.

Every valid EXP must introduce at least one genuinely new **hypothesis, model family, control formulation, dataset construction, evaluation protocol, or mechanism-level ablation** relative to the previous EXP. Merely changing a scalar coefficient, threshold, seed count, horizon by one step, cosmetic logging option, filename, or report wording does not qualify as a new experiment. Parameter sweeps and multiple settings of the same mechanism should normally be executed inside the same EXP.

`NOT_RUN_*`, `INTERFACE_GATE`, missing data, missing dependency, unsupported simulator feature, environment failure, build failure, package conflict, missing controller snapshot, missing checkpoint interface, insufficient disk layout, incomplete logging, or discovery that the current archive cannot answer the question **must not consume an EXP ID**. These are implementation problems inside the current experiment. You must solve them before the experiment is allowed to complete.

If such a problem is encountered during `EXP_Gk`, remain inside `EXP_Gk`. Diagnose it, inspect the relevant code and exact interface rather than guessing identifier names or paths, implement the missing collector or adapter, modify the simulator or controller integration when necessary, install or replace dependencies, create checkpoint/state-reset support, regenerate data, or redesign the experiment so that the same scientific question is genuinely tested. Use web research when necessary to inspect current papers, official documentation, repositories, implementations, algorithms, simulator APIs, or controller interfaces. Continue until `EXP_Gk` produces an actual `intervention -> feedback -> metric` chain.

A failed hypothesis is a valid EXP if the experiment itself genuinely ran. An unexecuted hypothesis is not a valid EXP.

## One-Gate Rule

The same data limitation, interface limitation, or infrastructure gate may consume **at most one valid EXP ID**, and only if that EXP produced new evidence before revealing the limitation. After the limitation is understood, the next work must pivot by changing the dataset construction, implementation, simulator interface, model family, control formulation, evaluation protocol, or mechanism being tested.

If a gate is discovered before any real experiment executes, it consumes zero experiment IDs. Repair the gate within the same EXP.

Do not repeatedly create reports saying that the same controller snapshot, state reset, dataset field, environment dependency, or simulator interface is missing. After the first discovery, fixing that limitation becomes part of the active experiment.

## Evidence Requirement

Every `EXP_G{id}` must leave machine-verifiable artifacts sufficient for an independent audit of what actually ran. The experiment directory must contain or reference the exact code revision, run command or configuration, random seeds when applicable, dataset manifest, train/development/held-out split definition, raw or minimally processed metrics, model checkpoint when a model is trained, intervention or rollout logs when control is executed, environment and dependency information, and evaluation outputs.

For simulator/controller studies, artifacts must make the causal chain reconstructable. At minimum, log the pre-intervention state or checkpoint identifier, active language action, proposed control or latent update, actual executed action, realized next robot state, next observation or observation identifier, re-encoded next latent, completion/switching state when relevant, and resulting metric.

Do not mark an EXP complete because a Python script terminated successfully. Completion requires scientific output and audit evidence.

Do not fabricate metrics, held-out evaluations, artifacts, model checkpoints, execution logs, statistical tests, or successful interventions. If a value cannot be traced to an artifact generated by an executed command, it may not appear as a reported result.

## Required Directory and File Discipline

Use a stable project structure. Keep experiment-specific raw artifacts and machine-readable results under an experiment directory such as `experiments/EXP_G{id}/`. Keep human-readable reports under `reports/`.

After a valid experiment has run, create `reports/EXP_G{id}_report.md`. This report should be as detailed as possible. It must explain the scientific hypothesis, why the experiment was necessary, what new mechanism or evidence it introduced, exact data used, exact implementation, training or fitting procedure, controller/simulator execution protocol, baselines, ablations, metrics, held-out protocol, repeated trials or seeds, quantitative results, qualitative failures, artifact locations, commands/configurations needed to reproduce the run, what was learned, what remains unresolved, and whether the result advances the final acceptance goal.

After the same experiment, also create `reports/next_exp_fromG{id}.md`. This file is the executable scientific prompt for the next iteration. It must derive the next experiment from the evidence produced by the current experiment. It must contain a **specific experiment that can materially advance the main research line and the final acceptance goal**. It may introduce a new model, new loss, new controller, new dataset construction, new causal intervention, new switching mechanism, new recovery formulation, or a mechanism-level ablation. It must not merely recommend further analysis or additional reporting.

The next-experiment file must include enough technical detail that execution can begin directly: the hypothesis, exact intervention, data required, proposed methods to compare, evaluation logic, machine-verifiable artifacts that must be produced, success/failure criteria, and how the result changes the next decision.

If the next experiment requires missing infrastructure, the prompt must explicitly instruct Codex to implement that infrastructure as part of the same experiment before scientific execution. It must never authorize creating a new EXP simply because infrastructure is missing.

## Self-Iteration Loop

Start with `EXP_G1`. Execute it fully. Produce `reports/EXP_G1_report.md` and `reports/next_exp_fromG1.md`. Then immediately read the report, inspect the generated artifacts, critically evaluate whether the result truly advances the paper, and execute the specific experiment defined in `next_exp_fromG1.md` as `EXP_G2`.

Continue the same loop for `EXP_G3`, `EXP_G4`, and onward. Do not wait for human confirmation between valid experiments unless an action is impossible because of external credentials, unavailable physical hardware, or another resource that cannot be created, simulated, downloaded, or replaced programmatically. Even in that case, first exhaust local code inspection, simulator alternatives, public documentation, public repositories, and technically valid surrogate implementations that preserve the scientific question.

After every experiment, perform a strict self-review before assigning the next ID. Ask whether new evidence was actually produced, whether the evidence is causal or merely surrogate, whether the winning method actually supports the stated hypothesis, whether a baseline unexpectedly falsifies part of the story, whether the final architecture should be simplified, and what single scientific bottleneck now most limits the end-to-end goal.

Do not protect earlier hypotheses for narrative consistency. If F1 is unnecessary, remove or redefine it. If a simple controller outperforms a complicated latent planner, investigate why. If learned F3 cannot match the required switching reliability, redesign the completion mechanism. If the latent representation provides no advantage over state-space control, treat that as a central scientific result rather than hiding it.

## EXP_G1 Required Starting Point

`EXP_G1` must attack the largest unresolved causal gap directly: **build and validate an action-conditioned closed-loop transition benchmark with recoverable simulator/controller checkpoints**.

The experiment must create a real loop in which a state can be restored, multiple control proposals can be executed from the same or sufficiently matched starting state, the simulator/controller produces the realized next state, the next observation is captured, and the action representation is re-encoded after the intervention.

Do not begin `EXP_G1` by inventing another offline latent planner. First inspect the existing repository, simulator/controller stack, datasets, wrappers, logging utilities, state-reset functions, controller state, robot state, observation pipeline, and action execution path. Use the exact identifiers and interfaces found in the code. Do not guess variable names, keys, paths, field names, or checkpoint formats.

If the current simulator lacks full checkpoint restore, implement the strongest reproducible state restoration mechanism allowed by the actual simulator. This may require restoring robot joints, velocities, object poses, object velocities, gripper state, controller internal state, task state, random state, and other simulator state. Determine the necessary fields from the real environment implementation and validation tests.

Validate restoration empirically. Restore the same checkpoint multiple times, apply the same action, and measure next-state reproducibility. Then restore the same checkpoint and apply multiple distinct actions or F2 proposals. Record the resulting next states and re-encoded latents. This establishes the causal data primitive required by every later F2 experiment.

`EXP_G1` should contain multiple implementation approaches when appropriate. For example, compare native simulator snapshot APIs against explicit state serialization if both are available; compare deterministic replay against controlled stochastic replay; compare full-state restoration against the smallest state representation that still reproduces the next-state distribution. These are implementation/mechanism comparisons inside one EXP, not separate EXP IDs.

The essential output of `EXP_G1` is a reusable dataset and interface for `current state + action/proposal -> realized next state + next latent`, together with restoration reproducibility metrics and audit logs. If the interface initially fails, remain in `EXP_G1` and fix it. `EXP_G2` is forbidden until this causal transition primitive actually works.

## Scientific Progression After EXP_G1

After the causal benchmark exists, the next experiments should be chosen from the bottleneck exposed by evidence, while preserving the final goal.

The most likely early direction is to isolate F2 with oracle F3. Compare strong control families from matched states using actual environment feedback. Possible families include local learned dynamics, residual dynamics, retrieval-conditioned control, latent graph planning, MPC-like shooting, CEM, MPPI, value-guided planning, uncertainty-aware ensembles, support-constrained planning, and direct state/action-space baselines. Several families should be compared inside the same EXP when they answer the same mechanism question.

Once a closed-loop F2 formulation produces reliable atomic execution, test **current-action protection** on an ordered pair such as `lift -> place`. Give the executive access to the next action while measuring whether future-goal information destabilizes the current lift. Compare unrestricted next-goal conditioning, hard gating, continuous authority gating, explicit active-action masks, source/target factorization, or other genuinely distinct control mechanisms within one EXP.

Then test two-step execution with oracle F3: complete `lift`, switch at the true boundary, and execute `place` from the physically reached current state. Compare current-state retargeting against restart-from-initial-state or open-loop baselines. Only after this works should learned F3 become the primary bottleneck.

For learned F3, evaluate temporal completion models as control mechanisms. Offline classification metrics may be reported, but the decisive metric is whether learned switching preserves current-action completion and downstream multi-action success in actual closed-loop rollout.

After robust autonomous two-step execution, extend to longer ordered compositions, perturbation tests, state distribution shifts, unseen action pairs where scientifically justified, and checkpoint-based recovery. Recovery claims must involve executed return/control from actual simulator states.

This progression is not a fixed list of experiment IDs. The evidence from each EXP determines the next one. A failed mechanism should trigger a scientifically meaningful pivot, not a cosmetic variant.

## Multi-Method Requirement Within Each EXP

Each EXP should be ambitious enough to answer a real scientific question. When feasible, test multiple model families, control strategies, objectives, data constructions, or causal ablations inside the same experiment. Do not use one EXP for `weight=0.1`, the next for `weight=0.2`, and the next for `weight=0.3`.

Hyperparameter tuning should occur inside the development portion of a single EXP. The held-out evaluation should remain protected and should be opened only after the design choice is fixed, unless the evaluation protocol explicitly requires nested cross-validation or repeated prospective trials.

A new EXP ID should correspond to a new scientific reason for running code.

## Use of Online Research

You are authorized and expected to use internet research whenever the current method stalls or a technical implementation is unclear. Search recent papers, official simulator documentation, source repositories, controller implementations, state-reset examples, MPC implementations, representation-learning methods, long-horizon manipulation methods, temporal completion models, uncertainty methods, and relevant benchmark protocols.

Prefer primary sources: papers, official documentation, and original repositories. Record the exact external method or implementation that influenced the experiment in the report. Reproduce or adapt methods accurately rather than relying on names alone.

Internet research is a tool for resuming execution. It may not become a substitute for execution. A literature summary without a new implemented and run experiment does not consume an EXP ID.

## Failure Handling

When a run fails because of code, infrastructure, simulator, or dependency issues, treat the failure as debugging inside the same experiment. Inspect logs and source code, fix the cause, add regression tests when appropriate, and rerun.

When the scientific hypothesis fails, preserve the result. Save the artifacts, quantify the failure, analyze the mechanism, and choose a different scientific direction for the next EXP.

When the available data is fundamentally incapable of answering the question, build the required data collection procedure and collect the data inside the active EXP. If physical hardware is genuinely unavailable, determine whether the scientific question can be tested in the simulator without changing the claim. If it cannot, narrow the claim explicitly and prioritize the strongest executable evidence rather than pretending that an offline surrogate proves physical behavior.

## Required Report Integrity Checks

Before completing every report, cross-check the prose against the raw artifacts. Verify that every winner is actually the best under the declared selection rule. Verify that a `SUPPORTED` conclusion logically matches the hypothesis. If the method proposed by the hypothesis loses to a baseline, the hypothesis is not supported.

Do not use vague statuses such as `SUPPORTED_STAGE` unless the report defines exactly what was supported and what was not. Prefer explicit statements such as `SUPPORTED: closed-loop F2 reduced realized endpoint error under oracle F3` or `NOT_SUPPORTED: learned F3 did not match oracle switching reliability`.

The report must distinguish direct evidence, inference, diagnostic surrogate evidence, and untested claims.

## Stop Conditions

The program stops successfully only when the final acceptance goal in `ACTIONS_AS_COORDINATES_FINAL_METHOD_AND_GOAL.md` has been achieved with auditable evidence.

At minimum, the system must demonstrate a closed-loop ordered two-action task such as `lift -> place` in which the current action remains stable despite knowledge of the future action, F2 decisions use action-conditioned simulator/controller feedback, switching is autonomous rather than oracle-driven in the final integrated run, the second action begins from the physically realized current state rather than the initial episode state, and the method achieves repeatable gains over appropriate open-loop, teacher-forced, restart-based, and module-ablated baselines.

The final evidence must also show that the learned action latent contributes meaningfully to the control result. If a non-latent baseline matches the full system, investigate and report that result before declaring success.

If the acceptance goal is reached before `EXP_G50`, stop the iterative program and write final synthesis artifacts that summarize the best system, supported claims, failures, ablations, reproducibility evidence, and remaining limitations.

If `EXP_G50` is reached without satisfying the final goal, stop creating new EXP IDs. Produce a final failure analysis that identifies which acceptance conditions remain unmet, the strongest evidence obtained, and the technically justified next direction. Do not claim project success.

## Final Behavioral Rule

The purpose of this program is experimental progress. A report is the record of work; it is never a substitute for work.

At every iteration, prefer the action that creates new causal evidence. If the environment blocks the experiment, fix the environment. If the dataset cannot answer the question, create the dataset. If the current model family fails, implement a different family. If the literature contains a stronger method, study and implement it. If an assumption is falsified, change the scientific model.

Do not advance from `EXP_Gk` to `EXP_G{k+1}` until `EXP_Gk` has generated real, machine-verifiable scientific evidence.
