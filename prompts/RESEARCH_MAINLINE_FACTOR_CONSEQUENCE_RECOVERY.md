# Research Mainline: Predict Only What Matters for Confirmed-Failure Robot Recovery

## Status and Authority of This Document

This document is the **non-negotiable research charter** for the next experimental phase. It defines the scientific question, the method boundary, the final objective, the claims that are intentionally abandoned, the order in which unresolved problems must be solved, and the evidence required before the paper can be considered complete.

Future experiments may change architectures, recovery proposal generators, data collection procedures, consequence factors, losses, selectors, controller wrappers, simulators, or evaluation protocols. They may not silently change the central scientific question. If one implementation family fails, replace that implementation while preserving the research mainline unless accumulated evidence directly falsifies the mainline itself.

The project begins **after a manipulation failure has already been confirmed**.

---

## The Core Story

A robot is executing a manipulation task with a base policy. A failure has already occurred and has been confirmed by an external mechanism. We do **not** study how to forecast that failure in advance. The robot now has one problem: **what should it do next to recover?**

A rich world model could evaluate each possible recovery by predicting a long future rollout: future images, future robot states, future object trajectories, future actions, or a high-dimensional latent trajectory. Our central hypothesis is that this may be more prediction than the recovery decision requires.

Instead, for each executable recovery proposal, we predict only a compact set of **decision-relevant, factorized action consequences**. The robot uses these predicted consequences to select one local recovery or to abstain and fall back to base-policy replanning. It executes only the selected recovery, observes the real resulting state, and decides again. When recovery has produced a state from which ordinary task execution can continue, control returns to the base policy.

The primary scientific question is:

> **Given a confirmed manipulation failure, are compact factorized action consequences sufficient to support broad, reliable, selected-only, closed-loop recovery decisions?**

Only after the complete recovery pipeline is strong do we ask the stronger question:

> **Does recovery actually require predicting a richer future world, or can the robot predict only the consequences that matter for the recovery decision?**

The second question is a final claim escalation, not a prerequisite for the first complete paper.

---

## Final Overall Objective

The final system must implement and validate the following causal loop:

```text
base policy executes task
        |
        v
confirmed failure event
        |
        v
generate executable recovery proposals
        |
        v
predict factorized consequences for each proposal
        |
        v
select exactly one local recovery OR choose fallback/replan
        |
        v
execute only the selected choice
        |
        v
observe the real post-action state
        |
        +---- recovered / handoff valid ----> return control to base policy
        |
        +---- still locally recoverable ----> generate/select another recovery
        |
        +---- no trustworthy local recovery -> safe fallback + base-policy replanning
```

The system must never execute every recovery proposal online before deciding which one to use. Counterfactual execution of all proposals is allowed only for offline training data generation, simulator oracle analysis, diagnostics, and evaluation after the deployed choice has been defined.

The final paper must show that the consequence representation is **decision-useful**, not merely predictive in isolation. The key outcome is whether the representation supports correct recovery or fallback decisions under closed-loop execution on unseen confirmed-failure states.

---

## Scope We Explicitly Abandon

### We Do Not Predict When the Robot Is About to Fail

Failure detection, impending-failure forecasting, early-warning prediction, trigger precision-recall optimization, change-point detection, and learned estimation of whether nominal execution will eventually fail are **outside the main contribution**.

The recovery method receives a confirmed failure event as an input condition. In simulation this may come from an environment predicate, benchmark-defined failure condition, controller-level failure flag, or deliberately constructed confirmed-failure checkpoint. In later real-robot evaluation it may come from a controller safety signal, external verifier, operator confirmation, or another upstream detector.

The source of the signal must be documented, but improving that detector is not this paper's goal.

No future experiment may reintroduce failure prediction as a prerequisite for validating the core recovery method.

### Active Nominal Probe Is Removed from the Main Method

Active nominal probing was useful in earlier experiments for studying observability and causal evidence. It is no longer a required mechanism for deciding whether failure exists.

A probe may be used later only if it is scientifically motivated as an auxiliary input for **post-failure consequence estimation**, and then it must be evaluated as an ablation. It may not become the main failure detector again.

### We Do Not Assume Every Failure Is Locally Recoverable

Some failure states have no viable short local intervention. Forcing the selector to choose a local recovery in such states is a formulation error.

Therefore **fallback is a first-class recovery decision**.

Fallback does not mean pretending that returning the arm to a home pose resets the world. The default fallback abstraction is:

> **safe local disengagement followed by base-policy replanning from the current observed world state.**

If an environment reset is required for a particular benchmark condition, it must be reported explicitly and separately. It may not be hidden inside the recovery method.

### A Fixed Recovery Route Library Is Not the Scientific Contribution

Previous experiments showed that proposal viability depends strongly on task and state. A small fixed route library cannot be assumed to cover manipulation failures broadly.

The proposal generator must be modular. Conceptually,

\[
\mathcal{R}(s,g,h) = G(s,g,h),
\]

where:

- \(s\) is the confirmed failure state available to the deployed system,
- \(g\) is the task goal,
- \(h\) is admissible execution/controller context,
- \(G\) is a proposal generator,
- \(\mathcal{R}\) is the set of executable local recovery proposals.

A proposal may be a short action chunk, retrieved corrective trajectory, analytic/task-space skill, semantic recovery option, learned local policy output, recovery subgoal executed by a frozen policy, MPC-generated short plan, or another executable object supported by the repository and simulator.

The proposal generator answers **“what could we try?”** The consequence model answers **“what is likely to happen if we try it?”** These two questions must remain experimentally separable.

---

## What We Preserve from Earlier Experiments

The new phase does not discard the positive evidence already obtained.

Earlier work established that factorized success/safety-related action-consequence information can contain useful pre-execution signal for recovery selection under matched same-state comparisons. The prior pipeline work also established a machine-verifiable selected-only causal chain in which a predicted recovery was selected before execution, only that recovery was executed, the environment was re-observed, and control returned to nominal execution before final task success.

At the same time, later experiments exposed structural weaknesses in early failure detection, fixed proposal coverage, selector calibration/generalization, and handoff-label consistency. The new phase is designed around those findings rather than trying to repair the old trigger formulation indefinitely.

Prior results are evidence and baselines. They must not be silently rewritten or tuned using consumed confirmation outcomes.

---

## The Scientific Object: A Factorized Consequence Interface

For a confirmed failure state \(s\) and an executable recovery proposal \(r\), the central predictive object is

\[
F_\theta(s,r) \rightarrow \mathbf{c}(s,r),
\]

where \(F_\theta\) is the learned consequence predictor and \(\mathbf{c}(s,r)\) is a compact vector of decision-relevant consequences.

The exact consequence factors must be justified by measured failure mechanisms and deployment semantics. They must not be invented only to make the representation appear richer.

Existing evidence strongly motivates **success-related consequence** and **unsafe consequence** as initial factors. Earlier handoff failures strongly motivate **handoff/continuation compatibility** as a high-priority hypothesis, but its exact definition must be validated in the new benchmark before it is treated as an established factor.

Additional factors such as task progress, recoverability, contact stability, or phase compatibility may be added only when data show that they correspond to a distinct failure mode or decision need.

The target is the **smallest empirically justified consequence set that supports the required recovery decisions**.

---

## Why Factorized Consequences Must Be More Than Multiple Output Heads

The paper cannot claim novelty merely because one scalar predictor was replaced by two or three classifiers. The project must establish that factorization creates a meaningful representation/interface.

At least several of the following properties must be demonstrated experimentally:

- **Distinct semantics:** each factor corresponds to a separately measurable property of the deployed recovery outcome.
- **Decision relevance:** changing or removing a factor changes which recovery is selected in states where that factor matters.
- **Mechanism-level ablation:** removing, merging, or scalarizing factors produces interpretable failure modes rather than an arbitrary accuracy change.
- **Generalization benefit:** factorization improves ranking, calibration, abstention/fallback behavior, safety, or cross-task/cross-proposal transfer relative to a matched scalar or direct selector.
- **Modularity:** the same frozen consequence representation can support more than one downstream decision rule without retraining the representation.
- **Compositionality across proposal families:** the interface remains meaningful when the proposal generator changes, rather than merely memorizing route identity.
- **Calibration or risk control:** factor-specific predictions permit explicit constraints or Pareto decisions that a single undifferentiated score cannot support as reliably.
- **Causal alignment:** labels correspond to the exact deployment sequence, including recovery duration, post-recovery observation, handoff, replanning, and termination semantics.

If a scientifically matched scalar verifier or direct selector performs equally well on all important dimensions, the paper must acknowledge that factorization may not be necessary and narrow the contribution accordingly.

---

## The Five Problems That Must Be Solved, in This Order

## 1. Build the Confirmed-Failure Benchmark

The first priority is to rebuild evaluation around **confirmed failure checkpoints** so that failure-detection quality is removed from the experiment.

Each benchmark unit must contain a reproducible robot/environment/controller state at which failure has already been established according to an explicit criterion. All compared methods must begin from the exact same restored failure state and use the same post-recovery semantics.

The benchmark must include both:

- states in which at least one useful local recovery exists, and
- states in which local recovery is unavailable or unreliable and fallback/replanning is the correct behavior.

The saved state must contain all variables required to reproduce deployment semantics. If pending action queues, controller memory, task phase, policy hidden state, or another controller variable changes outcome, that state must either be saved/restored exactly or deliberately cleared under a documented fresh-replanning protocol. Mixing these semantics silently is invalid.

The benchmark is not complete when only a list of checkpoints exists. It is complete when the checkpoints can be restored reproducibly, interventions can actually execute, outcomes can be measured, and matched method comparisons can be run.

## 2. Establish and Improve Recovery Proposal Coverage

Before optimizing a selector, determine whether the proposal set contains useful local recoveries.

For each confirmed failure state, simulator-only oracle analysis may restore the same state repeatedly and execute all legal recovery proposals. This yields **oracle recovery coverage**: the fraction of failure states for which at least one proposal produces a valid local recovery under the exact deployment handoff/replanning semantics.

This diagnostic separates two failure mechanisms:

```text
no viable proposal exists
versus
viable proposal exists but predictor/selector fails to choose it
```

If oracle coverage is low, do not spend experiment IDs tuning the selector. Improve the proposal generator using genuinely different proposal families.

If oracle coverage is high but learned recovery remains low, focus on consequence prediction, calibration, selector design, or control semantics.

The final paper must report proposal coverage separately from selection performance so that gains from better proposals are not misattributed to the consequence representation.

## 3. Define Deployment-Consistent Consequence Labels and Interface

The consequence target must represent the consequence that actually occurs online.

Every labeled recovery outcome must follow the same causal sequence used at deployment:

```text
confirmed failure state
    -> execute recovery proposal
    -> observe actual post-recovery state
    -> apply exact handoff / fallback / replanning semantics
    -> measure consequence outcome
```

Do not train labels using “recovery then fresh base policy” while deployment retains pending nominal actions. Do not train labels using one recovery chunk if deployment repeatedly executes the recovery before handoff. Do not score a state using a continuation policy different from the one used online.

The interface must explicitly define:

- what state/context the predictor receives,
- what exactly a recovery proposal contains,
- how long it executes and how it terminates,
- whether pending nominal actions are cleared or preserved,
- when the base policy is called again,
- what constitutes successful recovery,
- what constitutes an unsafe/worsening result,
- what constitutes valid handoff,
- when fallback is selected and what fallback physically executes.

A predictor cannot be expected to generalize if its target is not the deployed causal object.

## 4. Solve Selector Generalization on Unseen Confirmed Failures

Only after benchmark, proposal coverage, and consequence semantics are stable should the campaign optimize the learned decision system.

The selector must choose useful local recoveries or correctly abstain/fallback on **unseen confirmed-failure states**. Evaluation must distinguish:

```text
proposal-set has no solution
consequence prediction is wrong
consequence prediction is adequate but decision rule is wrong
selected recovery execution fails unexpectedly
handoff/replanning semantics fail
```

The main model must be compared against scientifically matched alternatives such as:

- direct state+proposal scalar verifier/value,
- direct classifier or direct route selector,
- success-only consequence prediction,
- safety-only filtering where meaningful,
- retrieval/route-prior or similarity-only controls,
- fixed/replan-only fallback baseline,
- factorized predictor with alternative downstream decision rules.

Model capacity, data, current-state information, proposal information, split protocol, and confirmation discipline should be matched as closely as practical.

The goal is not merely a high top-1 accuracy. The system must convert available proposal headroom into safe, useful recovery decisions under distribution shift.

## 5. Complete the Selected-Only Closed Loop

Once single-decision recovery is reliable, build the final stateful loop:

```text
confirmed failure
    -> proposals
    -> factorized consequence prediction
    -> one selected recovery or fallback
    -> execute
    -> reobserve
    -> if recovered: return to base policy
    -> if still locally recoverable: recover again
    -> if local recovery is not trustworthy: fallback + replan
```

The closed loop must log enough information to reconstruct every recovery cycle from machine artifacts. Re-observation must use the real resulting state, not an imagined state.

The paper should compare one-shot recovery against repeated short-horizon recovery with re-observation, and should test the value of explicit fallback/handoff logic when the data support those controls.

The final system is not complete until it is frozen and validated on a genuinely untouched confirmation cohort containing new demonstrations, initial conditions, failure states, and, where feasible, broader physical manipulation mechanisms.

---

## Fallback Is Part of the Policy, Not a Failure of the Method

The decision space should conceptually include

\[
\mathcal{A}_{\mathrm{rec}}(s) = \mathcal{R}(s) \cup \{\text{fallback/replan}\}.
\]

Fallback should be chosen when no local recovery is sufficiently trustworthy under the learned consequence interface and the predeclared decision rule.

A strong system should therefore succeed in two different ways:

- correctly executing a local recovery when a viable local recovery exists, and
- correctly refusing harmful or unsupported local interventions and handing the current world state back to a safe replanning path when local recovery is not justified.

The paper must measure both behaviors.

---

## Required Evaluation Decomposition

Aggregate end-to-end success alone is insufficient. Every final evaluation must be decomposable into at least:

```text
confirmed-failure benchmark validity
proposal oracle coverage
local-recovery opportunity rate
consequence prediction quality by factor
selection quality when a viable proposal exists
unsafe/harmful selection rate
fallback correctness / abstention quality
post-recovery handoff success
multi-cycle closed-loop recovery success
final task success after recovery
```

The exact metric names and formulas must be taken from the implemented environment and data structures. Do not invent field names, JSON keys, success predicates, unsafe predicates, or controller semantics from this document.

When uncertainty matters, use paired or clustered statistics that respect dependence between outcomes originating from the same demonstration, task, checkpoint, or episode.

---

## The Final `Only` Test Comes Last

The main recovery paper does **not** depend on beating a world model.

First complete a broad, reliable factorized-consequence recovery pipeline.

Only after that pipeline is stable should the project run a matched richer-future comparison. The comparison must use the same confirmed failure states, the same recovery proposals, comparable training data, comparable model capacity where practical, the same splits, and the same downstream decision evaluation.

A valid richer-future baseline must predict a materially richer representation of the future before producing recovery utility, for example future visual features, future state trajectories, future observation sequences, or another repository-supported high-dimensional rollout target.

Compare at least:

- final recovery/task success,
- harmful or unsafe selection,
- calibration / decision reliability where defined,
- robustness to task/state/proposal shift,
- compute and latency,
- prediction output dimensionality / prediction burden,
- data requirements,
- success-versus-risk decision frontier.

If the factorized system is competitive or superior while predicting substantially less future information, the title **Predict Only What Matters** is strongly supported.

If richer future prediction clearly improves recovery, report that result honestly. The work can still be complete under a narrower identity such as:

> **Factorized Action Consequences for Closed-Loop Robot Recovery**

Do not sacrifice the complete recovery contribution in order to force the word “Only.”

---

## Claims That Are Not Required and Should Be Dropped Unless New Evidence Demands Them

The project does not need to claim any of the following:

- prediction of impending failure,
- universal failure detection,
- active probing as a required recovery component,
- every failure is locally recoverable,
- one fixed recovery library covers arbitrary manipulation failures,
- one selector generalizes to every policy/task/domain,
- universal robot safety,
- replacement of all world models,
- policy-independent recovery,
- richer future prediction is unnecessary before a matched test proves it,
- recovery from physically irreversible terminal states,
- general-purpose open-world manipulation beyond the evaluated scope.

A smaller claim supported by clean causal evidence is preferred over a broad claim built on unsupported assumptions.

---

## What the Final Contribution Should Look Like

A successful paper should support a bounded chain close to the following:

> **For confirmed manipulation failures, a compact factorized action-consequence representation can serve as a decision interface for selecting among heterogeneous recovery proposals and fallback/replanning. The interface predicts recovery-relevant consequences before execution, supports selected-only closed-loop intervention with real re-observation, and enables the robot to return control to a base policy after recovery. The factorization is empirically meaningful rather than a cosmetic multi-head output, and its limitations are separated from proposal coverage and failure detection.**

The strongest version may add:

> **Under matched conditions, this compact consequence interface reaches a comparable or better recovery decision frontier than materially richer future prediction, showing that recovery can often be solved by predicting only what matters for the decision.**

The second statement is optional and must be earned by the final matched experiment.

---

## Literature and Method References for Codex

These references are **sources of mechanisms and baselines, not permission to copy their claims**. When a difficulty is encountered, inspect the primary paper and, when available, its official implementation. Borrow only the mechanism needed to address a specific observed failure mode. Preserve our own central claim: the scientific object is the factorized consequence interface for confirmed-failure recovery.

### RecoveryChaining: Learning Local Recovery Policies for Robust Manipulation

Primary paper: https://arxiv.org/abs/2410.13979

Useful idea: a distinct recovery control mode can return the system to a state where nominal controllers resume. It is useful for hierarchical recovery/handoff organization and for comparison against learned recovery-policy approaches.

Do not copy its contribution framing. Our focus is not learning a hierarchical recovery policy; our focus is predicting compact consequences of alternative recoveries and deciding among them or fallback.

### RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning

Primary paper: https://arxiv.org/abs/2409.14674

Useful idea: supervisor/actor decomposition, rich failure-recovery supervision, and language-conditioned correction. It may inspire proposal generation or failure-type-conditioned recovery options.

Do not drift into a VLM-supervisor paper unless data show that language supervision is necessary for proposal generation. Failure detection is outside our main scope.

### Fail2Progress: Learning from Real-World Robot Failures with Stein Variational Inference

Primary paper: https://arxiv.org/abs/2509.01746

Useful idea: generate failure-targeted data around observed failures rather than collecting generic data uniformly. This may inspire confirmed-failure benchmark expansion and local data generation around hard recovery states.

Do not replace our consequence-interface question with generic failure-data augmentation.

### Back to the Familiar Future: Failure Recovery for VLA Policies via Familiar Future Milestones

Primary paper: https://arxiv.org/abs/2606.09258

Useful idea: generate/select recovery milestones or familiar future anchors rather than relying on a fixed low-level route library. This is particularly relevant if proposal coverage is poor.

Use milestone/subgoal ideas as a **proposal generator**, not as a substitute for consequence prediction. The proposal generator and the consequence interface must remain separable.

### ReCoVLA: VLM-Guided Reward Compilation for Failure-Conditioned Residual Recovery

Primary paper: https://arxiv.org/abs/2606.09630

Useful idea: keep a pretrained VLA frozen while adding a failure-conditioned recovery layer and structured task-relevant reward components. It may inspire residual proposal families or structured consequence definitions.

Do not turn the project into VLM failure diagnosis or residual-policy fine-tuning unless that mechanism is required to produce executable proposals.

### PACTS: Jointly Learning Predicates and Actions Enables Zero-Shot Skill Composition

Primary paper: https://arxiv.org/abs/2605.20648

Useful idea: jointly model executable behavior with structured symbolic/predicate outcomes and expose those outcomes as a planning/monitoring interface. This is conceptually close to treating predicted outcomes as an interface rather than predicting only actions.

Our required distinction is recovery-specific counterfactual evaluation of alternative proposals and fallback under confirmed failure. Predicate interfaces may inspire factor definitions or compositional evaluation, but do not simply rename predicates as our factors.

### τ0-WM: A Unified Video-Action World Model for Robotic Manipulation

Primary paper: https://arxiv.org/abs/2606.01027

Useful idea: action-conditioned future visual rollouts and dense reward trajectories can evaluate candidate actions without physically executing all of them. This is a natural **richer-future baseline family** for the final `Only` test.

Do not attempt this comparison before the factorized recovery pipeline is stable.

### Inference-Time Enhancement of Generative Robot Policies via Predictive Foresight

Primary paper: https://arxiv.org/abs/2502.00622

Useful idea: an action-conditioned world model forecasts candidate-action consequences and performs lightweight online ranking/refinement. This is relevant both as a richer-future comparison and as a source of proposal-refinement ideas if coverage is poor.

Our difference must remain explicit: we study whether compact decision-relevant factorization can replace or reduce richer look-ahead for recovery.

### Feedback World Model Enables Precise Guidance of Diffusion Policy

Primary paper: https://arxiv.org/abs/2605.15705

Useful idea: close the loop between prediction and real post-action observation, correcting future decision-making after execution. This is relevant to our repeated `execute → reobserve → decide again` structure.

Our closed-loop recovery does not require maintaining a full world model; use this work as inspiration for feedback/re-observation controls and as a comparison point.

### EV-WM: Event-Verified World Models for Long-Horizon Robotic Manipulation

Primary paper: https://arxiv.org/abs/2606.13053

Useful idea: map imagined future visual features to task-grounded events such as object-state changes, progress, spatial relations, and success predicates. This is highly relevant to the boundary between rich future prediction and compact decision-level consequences.

If used as a baseline, preserve the causal distinction: EV-WM predicts a richer future first and then extracts task events, whereas our hypothesis is that recovery may predict the decision-relevant consequences directly.

### Foresight: Failure Detection for Long-Horizon Robotic Manipulation

Primary paper: https://arxiv.org/abs/2606.23085

Useful mainly as a **scope boundary**. It studies failure detection from action-conditioned predictive representations. If Codex encounters pressure to rebuild an early-failure monitor, read this literature to understand that failure detection is itself a substantial research problem, then return to our confirmed-failure formulation.

Do not drift into this topic.

---

## Literature-Use Rules

When external literature is used to solve a problem:

1. Search recent primary papers, official project pages, official repositories, and official documentation.
2. Record the exact paper/repository and the exact mechanism borrowed.
3. State which observed failure in our experiments motivated the borrowed mechanism.
4. Implement the mechanism as a testable component, not as a citation-only suggestion.
5. Compare it against simpler alternatives under the same confirmed-failure protocol.
6. Do not silently inherit another paper's claims, labels, task assumptions, or benchmark semantics.
7. Preserve our own claim boundary: **confirmed failure → proposal alternatives → factorized consequence prediction → selected-only local recovery or fallback → reobserve → base-policy return/replan**.

---

## Non-Negotiable Final Research Principle

The project is not asking:

> “Can we predict when the robot will fail?”

It is asking:

> **“The robot has already failed. What must it predict in order to choose what to do next?”**

The project succeeds when the repository contains a broad, auditable, frozen evidence chain showing that compact factorized consequences form a meaningful decision interface for closed-loop recovery, with proposal coverage, selector quality, fallback behavior, and handoff semantics measured separately.

Only after that is established should the project ask whether richer prediction of the future world adds anything essential.
