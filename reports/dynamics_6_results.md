# Eighteenth-wave results — CALVIN closed-loop reconstruction gate

## Outcome

**The planned closed-loop causal continuation study could not be executed because the retained public CALVIN artifacts do not permit exact reconstruction of source branch states.** The technical reconstruction-eligibility gate did not pass, so the closed-loop F1/F2 comparison was never run. This is not evidence that closed-loop refinement failed.

The six-task diagnostic replay covered 6 held-out validation annotations and 336 recorded transitions. Two simulators given the same approximate observation reset agreed exactly on all continuous state components and on terminal predicates (100.0%), but differed by one exposed contact point in every diagnostic. Thus the full-state median/P95 segment-maximum twin discrepancy was 1/1, above the frozen 1.0e-09 tolerance. Observed-reset replay also deviated from the recorded source, reaching raw robot/scene coordinate errors of 6.28208/1; the robot maximum includes Euler wrap and is reported only as a reconstruction diagnostic, not a physical metric. The unobserved source state cannot be recovered.

The official predicate was evaluated through `Tasks.get_task_info_for_set`. Approximate source replay reproduced the annotated terminal task on 5/6 diagnostics; `push_pink_block_right` was not reproduced. This is a gate diagnostic, not an estimate of task success.

The available held-out debug validation split has one authoritative continuous source session and eight language segments. Across all retained public data, exactly reconstructable source episodes = 0; required minimum = 180.

## Mandatory questions

1. **Reconstructable?** No. Approximate-reset twins match on continuous state and terminal predicates but not the exposed contact set; the source branch is not exactly reconstructable.
2. **Episodes/branches?** 0 eligible exact source episodes and 0 materialized branch points. Six segments were gate diagnostics only.
3. **Prospective selection?** Yes. First validation annotation/task was frozen before any wave-18 model output.
4. **Frozen?** Yes. Checkpoint hashes matched; no model was loaded, trained, or modified.
5. **Leakage?** No primary protocol ran. Diagnostic replay used future source actions only for the explicitly required zero-intervention gate, never as predictor input.
6. **F2 task success improvement?** Not tested.
7. **Paired episode-clustered CI?** Not computed; 0 eligible episodes.
8. **Breadth across tasks?** Not tested.
9. **H4 physical/decoded error?** Not tested.
10. **H8 physical/decoded error?** Not tested.
11. **Embodied off-manifold drift?** Not tested.
12. **F2 vs random norm-matched?** Not tested.
13. **F2 vs shuffled direction?** Not tested.
14. **Negative refinement degradation?** Not tested.
15. **Perturbation recovery?** Not tested.
16. **Perturbation basin?** Not estimated.
17. **Mechanism/outcome association?** Not tested.
18. **Failure modes repaired?** Not tested.
19. **Remaining failure modes?** Not classified because no model rollout was authorized.
20. **C4 closed-loop embodied refinement?** `NOT_TESTED_RECONSTRUCTION_GATE_FAILURE`.
21. **C5 learned direction causal value?** `NOT_TESTED_RECONSTRUCTION_GATE_FAILURE`.
22. **C6 proposal perturbation recovery?** `NOT_TESTED_RECONSTRUCTION_GATE_FAILURE`.
23. **Defensible story?** Wave-15/16/17 offline claims remain unchanged. Wave 18 adds no embodied claim.
24. **Further CALVIN work?** Yes, but only with prospectively stored exact branch snapshots and ≥180 independent source episodes.
25. **Most important experiment outside CALVIN?** A prospectively instrumented embodied domain that stores exact resettable state while using the already-frozen causal latent interface.

## Claim status

- C1/C2: unchanged, supported.
- C3a/C3b: unchanged, rejected.
- C3c-local: unchanged, strengthened by wave 16.
- C3c-long/C3d/context robustness: unchanged, supported by wave 17.
- C4/C5/C6: not tested because exact source branch reconstruction was unavailable.

No closed-loop refinement outcome direction was observed: neither success nor failure can be assigned to refinement from wave 18.

No expected or desired embodied conclusion was written in place of missing evidence.

Verification: **78 tests passed** across `tests/dynamics` and `tests/representation`.
