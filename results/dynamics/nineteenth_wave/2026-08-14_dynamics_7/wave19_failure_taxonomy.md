# Wave-19 failure taxonomy

## Source-data outcomes

- 335 π0.5 attempts were finalized: 306 official successes and 29 official horizon failures.
- 9 successful task-5 trajectories were too short for both 32 causal past steps and 128 future steps; they remain
  immutable raw successes but were not certified benchmark episodes.
- 297 successes had at least one eligible exact branch; 415 total 25%/50% branches passed certification.
- Task 8 reached 24 certified episodes at the frozen 50-attempt cap. This passed the 20 minimum and set the
  balanced primary scale to 24 per task (240 total).

## Resolved collection-system diagnostics

- The first task-0 branch exposed a restore-order error: `mj_forward` rewrote warm-start state. Restoring the full
  integration state again after forward made the immutable raw continuation exact at zero error.
- Task01 attempt025 exposed a predicate-read-order mismatch around a transient contact. Matching the source order
  `step -> check_success -> snapshot` produced zero predicate mismatches and exact recertification.
- One invalid recovery launch disabled Numba JIT and changed controller numerics. It was quarantined; the same raw
  episode passed at zero error under the original formal JIT mode.

## Scientific stop

- The six-seed representation semantic tests passed strongly, and gripper fidelity passed.
- Continuous motor fidelity missed the preregistered ratio by `0.000444393` in ratio units. The R-gate therefore
  failed; no selected representation, F1, F2, offline dynamics, B0–B5, or proposal-recovery result exists.
