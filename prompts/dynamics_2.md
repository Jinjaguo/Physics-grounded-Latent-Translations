# PGLT Fourteenth-Wave Codex Prompt — DEL Failure Adjudication

## Scope

This is a frozen-dynamics diagnostic wave.

Do not modify or retrain the representation. Do not collect longer trajectories yet. Do not introduce a new dynamics architecture. Do not tune on already-observed official validation results.

Wave 13 established:
- learned autonomous latent prediction beats copy;
- matched refinement is best in Block A;
- history MLP is best in Block B;
- unforced DEL is worse than MLP;
- forced DEL is worse than history MLP;
- DEL does not reduce decoded-action error or off-manifold drift;
- current evidence does not support variational inductive bias;
- DEL solvers are finite but strict residual convergence rate is 0;
- horizons 4/8/16 are unsupported by current annotation lengths.

The purpose of wave 14 is to determine **why DEL failed** before deciding whether longer trajectories should continue testing DEL or focus on non-variational latent dynamics.

Read first:
- `dynamics_1_results.md`
- `thirteenth_wave_results.md`
- `thirteenth_wave_next_experiment.md`
- `held_out_dynamics_evaluation.json`
- dynamics development evaluation
- dynamics confirmation manifest
- unforced/forced DEL checkpoints
- MLP/history-MLP checkpoints
- matched-refinement implementation
- corrected local LaWM solver adapter
- solver residual traces
- dynamics split preregistration
- frozen latent serialization manifest
- causal-control audit
- relevant tests

Artifacts and source are authoritative. Do not guess paths, coefficients, residual formulas, checkpoints, or timing.

## 1. Scientific question

Distinguish:

### A. Solver-limited DEL
The learned DEL residual is compatible with ground-truth transitions, but the current solver fails to recover the relevant root.

### B. Variational-model mismatch
Ground-truth transitions do not satisfy the learned DEL residual well, or low-residual roots lie far from the true next latent.

### C. Conditioning / basin problem
A useful root exists near the data transition, but the residual Jacobian is ill-conditioned or causal initialization enters the wrong basin.

### D. Mixed unforced/forced failure
Unforced and forced DEL fail for different reasons.

Do not write the diagnosis in advance.

## 2. Freeze every learned model

Do not retrain:
- representation
- MLP
- matched refinement
- unforced DEL
- history MLP
- forced DEL

Do not change split, latent serialization, context, force model, mass/potential networks, or Lagrangian parameters.

Any new root solver is diagnostic only.

## 3. Use development data for solver decisions

Official validation results have already been observed.

Do not use validation to choose solver type, iteration budget, tolerance, damping, or initialization.

Freeze all diagnostic settings before development evaluation.

The same frozen diagnostic may later be reported on validation descriptively, but not as a new held-out test.

## 4. Reproduce the exact DEL residual

Use the exact trained residual from source.

Unforced:
`R = D2 Ld(q_prev,q_cur;c) + D1 Ld(q_cur,q_next;c)`

Forced:
use the exact trained causal-history force terms and sign convention from source.

Add a regression test proving the diagnostic residual matches the wave-13 solver residual numerically.

## 5. Ground-truth residual compatibility audit

For every development transition evaluate residual at:
- ground-truth `q_next`
- historical DEL prediction
- MLP prediction
- history-MLP prediction where appropriate
- matched-refinement prediction

Report for unforced and forced DEL:
- L2 norm
- RMS per latent dimension
- mean/median
- p90/p95/p99
- per-task distribution
- per-episode distribution

Primary question:
**Does the true next latent have lower DEL residual than non-variational predictions?**

## 6. Residual-vs-error relationship

For every transition compute:
- DEL residual norm
- next-latent MSE
- decoded-action MSE
- off-manifold distance

Report rank correlations.

Question:
**Does lower DEL residual actually correspond to better prediction?**

If residual falls while prediction worsens, the learned variational field is misaligned with observed dynamics.

## 7. Frozen solver-budget diagnostic

Preregister iteration budgets:
`4, 8, 16, 32`

Use the exact current corrected iterative update and the same causal initialization as wave 13.

For every budget report:
- final residual norm
- latent MSE
- decoded-action MSE
- finite/nonfinite
- residual trace
- step-norm trace

Do not add more budgets after seeing results.

## 8. Robust root-solver diagnostic

Implement one deterministic diagnostic root solver over `q_next` only.

Preferred:
- damped Newton / Levenberg-Marquardt least-squares

If unavailable, use LBFGS on:
`0.5 * ||R||^2`

Freeze before evaluation:
- max iterations
- tolerance
- damping / line-search
- stopping rule

Do not tune on official validation.

Report:
- convergence rate
- residual
- latent MSE
- decoded-action MSE
- off-manifold distance

## 9. Multiple initialization adjudication

Use these preregistered initializations where available:

Causal:
1. historical DEL initialization
2. copy `q_current`
3. constant velocity
4. MLP prediction for unforced DEL
5. history-MLP prediction for forced DEL

Oracle/local-solvability diagnostic:
6. ground-truth next latent plus fixed small deterministic perturbation

The ground-truth-near initialization is diagnostic only and excluded from primary rollout claims.

Interpretation:
- GT-near converges but causal inits fail → basin/conditioning problem
- GT-near also fails → current residual field is locally incompatible with data

## 10. Root proximity audit

For every converged root report:
- distance to true next latent
- distance to MLP/history-MLP prediction
- nearest-training-latent distance
- decoded-action error
- semantic retention

A low-residual root is not useful if it is far from ground truth or off-manifold.

## 11. Residual Jacobian conditioning

For a deterministic development subset compute:
`J = dR/dq_next`

Evaluate at:
- true next latent
- historical DEL prediction
- converged robust root

Report:
- singular values
- min/max singular value
- condition number
- effective rank
- fraction nearly singular

Use a preregistered epsilon.

Do not interpret J as physical stiffness.

## 12. Local root multiplicity / basin audit

On a small deterministic subset, run robust solver from several fixed perturbations around:
- q_current
- MLP prediction

Use a preregistered perturbation scale based on dynamics-training latent std.

Cluster converged roots by latent distance.

Report:
- number of distinct roots
- residual of each
- distance to ground truth
- initialization→root mapping

Many low-residual roots far from data indicate underconstrained or ambiguous DEL dynamics.

## 13. Compare unforced and forced DEL

Repeat the full adjudication for both.

Forced DEL must use the exact causal history packet from wave 13. Never use future target actions.

Question:
**Does causal forcing make true transitions more compatible with DEL residual?**

If not, the current force formulation does not rescue variational compatibility.

## 14. Matched-refinement control

Use the frozen matched-refinement baseline to separate:
- benefit of iterative computation
- benefit of variational structure

Audit latent MSE, decoded-action error, and off-manifold drift.

Do not retrain.

## 15. Decision rules

### Outcome A — Solver bottleneck
Support only if:
- true transitions have relatively low DEL residual;
- robust causal solver reaches low residual;
- better convergence materially improves latent and decoded-action error;
- roots stay near the data manifold.

Next:
rerun equal-information dynamics with corrected solver on newly exposed longer trajectories.

### Outcome B — Variational-model mismatch
Support if:
- true residual is not low relative to alternatives, OR
- lower residual does not improve prediction, OR
- low-residual roots remain far from ground truth/off-manifold.

Next:
stop treating DEL as primary; collect/expose longer annotation-consistent trajectories and compare MLP vs matched refinement as the main long-horizon models. Keep DEL only as negative/diagnostic baseline.

### Outcome C — Conditioning/basin failure
Support if:
- useful root exists near ground truth;
- GT-near initialization converges;
- causal initializations fail;
- Jacobians are ill-conditioned or basin-sensitive.

Next:
preregister one solver/preconditioning intervention, then test prospectively on longer trajectories.

### Outcome D — Mixed controlled/uncontrolled result
Use only if unforced and forced DEL have clearly different mechanisms.

### Outcome E — Unresolved
Do not collect a large new dataset until ambiguity is documented.

## 16. Representation remains frozen

Regardless of outcome:
- representation remains frozen
- R-Gate remains PASS
- historical Gate A remains preserved
- EMA remains frozen

Do not reopen representation search.

## 17. Longer-trajectory collection decision

Do not collect automatically in this wave.

At the end:
- solver bottleneck → longer data should include corrected DEL
- model mismatch → longer data should focus on MLP vs matched refinement, DEL negative baseline only
- unresolved → define one more diagnostic before expensive collection

## 18. Tests

Add/preserve:
- frozen representation hashes
- frozen dynamics model hashes
- exact residual reproduction
- no learned-model parameter updates
- solver optimizes only q_next
- future-action exclusion
- robust solver finite behavior
- residual/Jacobian finite behavior
- GT-near init labeled oracle-only
- validation excluded from solver-setting selection
- matched-refinement unchanged
- LaWM corrected toy regression

Do not modify third-party CALVIN/LaWM repos. Do not overwrite wave-13 artifacts.

## 19. Deliverables

Produce:
- `fourteenth_wave_results.md`
- `fourteenth_wave_next_experiment.md`
- frozen model/checkpoint audit
- exact DEL residual regression test
- ground-truth residual compatibility table
- residual-vs-error analysis
- solver-budget report
- robust root-solver report
- initialization/basin report
- root-proximity report
- Jacobian-conditioning report
- root-multiplicity diagnostic
- unforced-vs-forced failure comparison
- matched-refinement interpretation
- final DEL failure-mechanism JSON
- longer-trajectory next-experiment decision
- commands/provenance/files-changed/tests
- updated `RESEARCH_LOG.md`
- updated `NEXT_EXPERIMENT.md`

Final report must answer:
1. How large is DEL residual at the true next latent?
2. Is ground truth lower-residual than MLP/history-MLP predictions?
3. Does increasing solver iterations reduce residual?
4. When residual decreases, does prediction improve?
5. Can a robust root solver converge where the historical solver does not?
6. Are converged roots close to the true next latent?
7. Does a ground-truth-near initialization reveal a valid local root?
8. Are residual Jacobians ill-conditioned?
9. Are there multiple low-residual roots?
10. Do unforced and forced DEL fail for the same reason?
11. Does matched refinement confirm iterative computation helps independently of DEL?
12. Is failure best explained by solver limitations, conditioning, or variational-model mismatch?
13. Should DEL remain a primary hypothesis in the next long-horizon experiment?
14. Should the next wave collect/expose longer trajectories, and which models should be compared?

Do not write the desired conclusion in advance.
Scientific correctness and solver auditability are the priority.
