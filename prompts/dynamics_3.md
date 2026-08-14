# PGLT Fifteenth-Wave Codex Prompt — Factorized Variational Dynamics on the Executable Subspace

## Purpose

This wave is the final preregistered attempt to test a **positive variational-dynamics claim** without reopening the frozen representation.

The full 32-D DEL hypothesis has already failed and must remain recorded as a negative result.

Wave 14 established on development data that both unforced and forced full-latent DEL suffer from **variational-model mismatch**:

- the true next latent does not have lower DEL residual than the corresponding MLP prediction;
- increasing solver iterations drives residual nearly to zero but does not improve prediction;
- for forced DEL, prediction error worsens as residual is reduced;
- converged roots remain far from the true next latent;
- Jacobian singularity is not the main explanation;
- generic matched refinement helps without requiring DEL structure.

Do not attempt to rescue the old full-latent DEL by further solver tuning.

This wave tests a revised hypothesis:

> The frozen action representation contains semantic and executable factors with different dynamical roles. Language/context anchors the semantic factor, while structured variational dynamics should act only on the executable factor. The executable factor should inherit its local geometry from the frozen action decoder rather than from a completely free latent-space metric.

The experiment must be able to reject this hypothesis cleanly.

If this wave fails, DEL must stop being a positive main-paper hypothesis for the current submission.

---

## 1. Preserve the Existing Representation Decision

The representation remains permanently frozen.

Do not modify:

- epoch-40 EMA representation checkpoints;
- 32-D total latent;
- 16-D `z_sem`;
- 16-D `z_exec`;
- action encoder;
- decoder;
- shared-trunk CLIP isolation;
- text tower;
- text projection;
- normalization;
- representation training data;
- R-Gate decision.

Representation optimizer steps must remain exactly zero.

Historical strict Gate A remains FAIL.
Prospective R-Gate remains PASS.
Representation-ready remains TRUE.

Do not rewrite these facts.

---

## 2. Preserve the Full-Latent DEL Negative Result

The following wave-13/14 conclusions are immutable:

```text
full-latent unforced DEL: not supported
full-latent forced DEL: not supported
full-latent variational_model_mismatch: supported
```

Do not retrain those models to make them look better.

Retain them only as frozen historical negative baselines.

The new experiment is scientifically distinct:

```text
full latent dynamics
vs
factorized executable-subspace dynamics
```

---

## 3. Revised Scientific Hypothesis

Let the frozen action latent be

\[
z_k = [s_k,e_k]
\]

where

- \(s_k=z_{\mathrm{sem},k}\in\mathbb R^{16}\)
- \(e_k=z_{\mathrm{exec},k}\in\mathbb R^{16}\)

The revised hypothesis is:

1. `z_sem` carries task/semantic anchoring and may evolve slowly or predictably under language context.
2. `z_exec` carries most of the directly useful execution variation.
3. A mechanics-like transition should constrain `z_exec`, not force the full semantic+execution vector to satisfy a single DEL law.
4. The local geometry used by the variational model should be tied to the frozen decoder's executable sensitivity.

The desired positive result is:

```text
execution-subspace DEL
    beats an execution-subspace MLP
    and a matched generic refinement baseline
    under the same information set,
    while reducing decoded-action error and off-manifold drift.
```

One-step residual reduction alone is not sufficient.

---

## 4. Use the Existing Dynamics Split

Reuse the exact wave-13 dynamics:

- training episodes;
- development episodes;
- official validation one-shot split;
- non-overlapping H=16 latent windows;
- stride=16;
- causal timing;
- language context.

Do not resample train/development episodes.

Do not inspect official validation to choose the new method.

All method and solver decisions must be made using dynamics training/development only.

Create a new frozen wave-15 preregistration before training.

---

## 5. Shared Semantic Predictor

To isolate the execution-dynamics question, train exactly one semantic predictor:

\[
\hat s_{k+1}
=
F_{\mathrm{sem}}(s_{k-1},s_k,c_k).
\]

Use a modest MLP.

This semantic predictor is shared by ALL new factorized dynamics models.

It must not be retrained separately per execution model.

The final predicted latent for every factorized model is:

\[
\hat z_{k+1}
=
[\hat s_{k+1},\hat e_{k+1}].
\]

This ensures that differences in full-latent and decoded-action performance arise from the execution transition, not from different semantic predictors.

Report parameter count separately.

Also report the semantic predictor's standalone MSE.

---

## 6. Block F — Equal-Information Execution Dynamics

All primary factorized execution models receive exactly:

\[
(e_{k-1},e_k,s_k,c_k)
\]

and no future target actions.

Use the same semantic predictor from Section 5.

Train/evaluate:

### F1. Execution MLP

\[
\hat e_{k+1}
=
F_{\mathrm{exec}}(e_{k-1},e_k,s_k,c_k).
\]

This is the primary unconstrained baseline.

### F2. Execution matched refinement

Use the same generic refinement idea that succeeded in wave 13, but operate only on `z_exec`.

It must receive the same information as F1.

Use matched inference iterations relative to the new variational model.

Do not introduce future action information.

### F3. Execution-only free DEL ablation

Use the same general DEL family as the failed full-latent model, restricted to the 16-D execution subspace:

\[
L_d^{\mathrm{free}}
(e_k,e_{k+1};s_k,c_k).
\]

This ablation answers:

**Was full-latent mismatch caused mainly by applying DEL to semantic dimensions?**

Do not make this the main proposed method.

### F4. Decoder-geometry execution DEL — PRIMARY VARIATIONAL MODEL

This is the only new structured method.

Use a discrete Lagrangian on the executable subspace whose kinetic metric is induced by the frozen decoder.

---

## 7. Frozen Decoder-Induced Executable Metric

Let the frozen decoder be

\[
a = D(s,e).
\]

At a local point \((s,e)\), define the execution Jacobian:

\[
J_e(s,e)
=
\frac{\partial D(s,e)}{\partial e}.
\]

Define a positive semidefinite pullback metric:

\[
G_e(s,e)
=
J_e(s,e)^\top
W_a
J_e(s,e)
+
\epsilon I.
\]

Use a fixed action-space weight matrix \(W_a\).

Default:

```text
W_a = identity over normalized continuous decoder outputs
```

Handle the gripper channel carefully:

- do not differentiate through a discrete threshold;
- use the decoder's continuous pre-threshold/logit output if available;
- otherwise exclude the gripper dimension from `W_a` for the metric and report this explicitly.

Do not choose action weights by validation performance.

Preregister `epsilon`.

The metric is not claimed to be a physical robot inertia matrix.

Interpret it as:

**the local executable geometry induced by how latent perturbations change decoded robot actions.**

---

## 8. Efficient Metric Computation

Use automatic differentiation.

You may compute:

- the exact Jacobian if tractable;
- or Jacobian-vector products sufficient to evaluate \(v^\top Gv\).

Prefer JVP/VJP if exact Jacobian materialization is unnecessarily expensive.

Add tests that the metric quadratic form is non-negative up to numerical tolerance.

Do not backpropagate into decoder parameters.

Decoder parameters remain frozen.

---

## 9. Decoder-Geometry Discrete Lagrangian

Define midpoint quantities:

\[
\bar e_k = \frac{e_k+e_{k+1}}{2},
\qquad
v_k = \frac{e_{k+1}-e_k}{h}.
\]

Use:

\[
T_d
=
\frac12
v_k^\top
G_e(\bar s_k,\bar e_k)
v_k.
\]

For the primary model, use:

\[
\bar s_k = s_k
\]

because context is fixed within the annotation window.

Use a learned scalar potential:

\[
V_\theta(e_k,s_k,c_k).
\]

Then:

\[
L_d^{\mathrm{geom}}
=
h(T_d - V_\theta).
\]

Only the potential network is trainable if the metric is fully decoder-induced.

Do not introduce a free full mass matrix.

---

## 10. DEL Transition

The execution DEL residual is:

\[
R_e
=
D_2L_d(e_{k-1},e_k;s_k,c_k)
+
D_1L_d(e_k,e_{k+1};s_k,c_k).
\]

Predict \(e_{k+1}\) via the corrected differentiable root/refinement solver.

Preferred causal initialization:

```text
execution MLP prediction
```

This is allowed only if the matched-refinement baseline uses the same initialization and same refinement-step budget.

This creates the clean test:

```text
generic refinement of MLP prediction
vs
variational refinement of the same MLP prediction
```

---

## 11. Matched Refinement Fairness

F2 and F4 must use:

- identical initial execution MLP prediction;
- identical number of refinement iterations;
- comparable step/line-search budget;
- identical causal information.

F2 optimizes a generic learned/local consistency objective with no DEL structure.

F4 optimizes the decoder-geometry DEL residual.

Record wall-clock and iteration cost.

The paper claim can only attribute benefit to variational structure if F4 exceeds F2.

---

## 12. Training Objectives

### F1 Execution MLP

\[
\mathcal L_{\mathrm{exec}}
=
\|\hat e_{k+1}-e_{k+1}\|^2.
\]

### Shared semantic predictor

\[
\mathcal L_{\mathrm{sem}}
=
\|\hat s_{k+1}-s_{k+1}\|^2.
\]

### F3/F4 DEL

Primary training objective must optimize next-state quality through the differentiable transition:

\[
\mathcal L_{\mathrm{pred}}
=
\|\hat e_{k+1}-e_{k+1}\|^2.
\]

A DEL residual term may be added only with one fixed preregistered coefficient.

Do not select the coefficient using official validation.

Do not train solely by residual minimization.

---

## 13. Compatibility Preflight Before Full Evaluation

Before official validation, run a wave-14-style compatibility audit on dynamics development.

For F3 and F4 evaluate DEL residual at:

- true next execution latent;
- execution MLP prediction;
- matched-refinement prediction;
- DEL prediction.

The new variational hypothesis is allowed to proceed only if F4 satisfies BOTH:

### Compatibility criterion

Mean true-next residual is lower than the mean residual at the execution-MLP prediction.

### Alignment criterion

At minimum require:

```text
Spearman(residual_norm, execution_MSE) > 0
```

because larger residual should correspond to larger prediction error.

Report uncertainty if practical.

If F4 fails these tests strongly, the positive DEL claim is rejected.

---

## 14. Development Metrics

Evaluate each factorized model on:

### Execution state
- one-step `z_exec` MSE;
- cosine similarity;
- two-step rollout `z_exec` MSE.

### Full latent
Using the shared semantic predictor:
- full 32-D latent MSE;
- semantic MSE;
- execution MSE.

### Decoded action
Decode `[predicted z_sem, predicted z_exec]` through the frozen decoder:
- continuous action MSE;
- gripper accuracy;
- all seven action dimensions.

### Semantic retention
- predicted latent→text retrieval;
- semantic cosine to task prototype.

### Off-manifold
Report separately:
- full-latent kNN radius;
- execution-subspace kNN radius;
- distance to training `z_exec` manifold;
- fraction beyond preregistered train quantile.

### Structured diagnostics
For F3/F4:
- residual norm;
- convergence;
- solver iterations;
- nonfinite rate.

---

## 15. Primary Development Ranking

Primary comparison:

```text
F1 execution MLP
F2 execution matched refinement
F4 decoder-geometry execution DEL
```

F3 free execution DEL is an ablation.

The historical full-latent MLP/matched-refinement/DEL results remain contextual baselines.

Primary score on current data:

```text
supported-horizon normalized rollout AUC over horizons 1 and 2
```

Secondary required metrics:

- one-step execution MSE;
- two-step execution MSE;
- decoded-action MSE;
- off-manifold drift;
- semantic retention.

---

## 16. Hard Positive-Claim Gate for Variational Dynamics

A positive variational claim is authorized only if F4 satisfies ALL on dynamics development:

1. F4 beats F1 execution MLP on rollout AUC.
2. F4 beats F2 matched refinement on rollout AUC.
3. F4 does not materially worsen one-step decoded-action MSE.
4. F4 lowers execution-subspace off-manifold drift relative to F1.
5. F4 passes the residual compatibility criterion.
6. F4 passes the residual-error alignment criterion.
7. Solver nonfinite rate = 0.
8. Solver convergence is materially above the historical full-latent DEL convergence rate of 0.

Do not relax this gate after seeing results.

If F4 only beats F1 but not F2, the result supports refinement, not variational structure.

---

## 17. One-Shot Official Validation

Freeze all new checkpoints and settings using dynamics development only.

Write:

`factorized_dynamics_confirmation_manifest.json`

before reading official validation metrics.

Then run one-shot official validation.

Report:

- F1
- F2
- F3
- F4

with the same metrics.

Do not tune afterward.

The positive variational claim requires the development gate and validation direction to agree.

---

## 18. Long-Horizon Decision

Do not collect/expose longer trajectories until this wave is complete.

### If F4 passes

Next wave:
Expose/collect annotation-consistent trajectories with at least 10 non-overlapping H=16 windows per valid segment when possible.

Prospectively compare at horizons 1/2/4/8:
- factorized execution MLP;
- factorized matched refinement;
- decoder-geometry execution DEL.

The old full-latent DEL remains a negative baseline.

### If F4 fails but F2 wins

Revise the main dynamics claim.

Next long-horizon experiment focuses on:

```text
MLP vs structured/generic refinement
```

DEL remains a negative mechanistic baseline.

Do not perform another DEL rescue for the current paper.

### If neither F2 nor F4 beats F1

The evidence supports predictive latent coordinates but not structured long-horizon dynamics.

Do not claim stable structured dynamics.

---

## 19. Paper-Claim Decision After This Wave

Write a machine-readable claim decision.

### C1 — language-addressable action coordinate
Remain supported.

### C2 — executable continuous action coordinate
Remain supported.

### C3a — full latent obeys useful DEL dynamics
Set permanently to:

```text
REJECTED
```

### C3b — executable subspace admits useful decoder-grounded variational dynamics
Set after wave 15 to:

```text
SUPPORTED
or
REJECTED
```

based on the hard gate.

### C3c — generic structured refinement improves latent prediction
Evaluate independently from F2.

---

## 20. Allowed Paper Story If F4 Succeeds

If F4 passes, the paper story may be:

> A language-grounded action latent should not be treated as one homogeneous mechanical coordinate. Its semantic and executable factors play different roles: language anchors the semantic component, while decoder-grounded variational dynamics constrains the executable component.

Suggested one-line English story:

> **Language anchors what an action means; variational executable geometry constrains how it evolves.**

Suggested Chinese:

> **语言锚定动作 latent 的语义成分，而由可执行解码几何约束的变分动力学负责其执行成分的连续演化。**

Do not claim all latent dimensions obey physical mechanics.

---

## 21. Required Paper Story If F4 Fails

If F4 fails the hard gate, do not keep DEL as a positive paper claim.

The paper may instead claim:

> Language-grounded continuous action coordinates are simultaneously semantically addressable and executable, and support learned latent prediction. Generic refinement can improve local transition quality, while naive discrete variational mechanics is mismatched to the learned representation.

Possible one-line story:

> **Language defines meaningful action coordinates; structured refinement keeps learned transitions near executable regions.**

Do not phrase this as evidence for variational mechanics.

---

## 22. Tests

Add/preserve tests for:

- frozen representation hashes;
- exact 16/16 latent slicing;
- shared semantic predictor identity across all factorized models;
- decoder parameter gradients = 0;
- decoder Jacobian correctness by finite difference on a deterministic sample;
- decoder-induced metric PSD within tolerance;
- no future target-action access;
- identical F2/F4 initialization;
- identical refinement iteration counts;
- exact dynamics split;
- development-only method selection;
- official validation manifest before test;
- residual compatibility computation;
- execution-subspace off-manifold metric;
- no representation updates;
- historical full-latent DEL artifacts unchanged.

Do not modify third-party CALVIN or LaWM repositories.

Create a timestamped wave-15 dynamics directory.

---

## 23. Required Deliverables

Produce:

- `fifteenth_wave_results.md`
- `fifteenth_wave_next_experiment.md`
- factorized-dynamics preregistration
- frozen representation/model audit
- semantic predictor specification/results
- execution MLP specification/results
- execution matched-refinement specification/results
- execution-only free DEL ablation
- decoder-geometry DEL specification
- decoder-Jacobian/metric validation report
- parameter-count table
- compatibility-preflight report
- residual-vs-error report
- development rollout metrics
- decoded-action metrics
- semantic-retention metrics
- full-latent and execution-subspace off-manifold metrics
- factorized dynamics confirmation manifest
- one-shot official validation results
- hard variational-claim gate JSON
- paper-claim decision JSON
- exact commands
- environment/provenance
- files changed
- updated tests
- updated `RESEARCH_LOG.md`
- updated `NEXT_EXPERIMENT.md`

The final report must explicitly answer:

1. Does restricting dynamics to `z_exec` remove the residual mismatch seen in the full 32-D latent?
2. Does the decoder-induced metric make the true next execution latent lower-residual than the execution-MLP prediction?
3. Is DEL residual positively aligned with execution prediction error?
4. Does execution-only free DEL outperform historical full-latent DEL?
5. Does decoder-geometry execution DEL outperform execution MLP?
6. Does it outperform matched generic refinement under identical initialization and iteration budget?
7. Does it reduce execution-subspace off-manifold drift?
8. Does the hybrid `[predicted z_sem, predicted z_exec]` decode more accurately?
9. Does semantic retention remain stable?
10. Does the solver converge materially better than historical full-latent DEL?
11. Does one-shot validation preserve the development ordering?
12. Is C3b supported or rejected?
13. Is generic structured refinement supported independently of DEL?
14. Which exact paper story is now scientifically defensible?
15. Should the next wave expose longer trajectories, and which models should be carried forward?

Do not write the desired conclusion before seeing results.

This is a falsifiable final test of the variational-mechanics claim for the current paper.
