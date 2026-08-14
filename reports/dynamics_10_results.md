# Twenty-second wave results: executable coordinate consistency

Run date: 2026-08-14T17:50:14.746264-04:00

## Outcome

- M0 decoder-consistency mechanism: **REJECTED**
- C9 executable language redirect: **NOT_TESTED**
- C10 language as executable target coordinate: **NOT_TESTED**
- Wave22 optimizer steps: **0**

A1-A4 passed, but A5 failed prospectively. Four frozen cycle-map iterations strongly reduced cycle residual while full RedirectGain lost a strictly positive clustered lower bound and six-way endpoint accuracy decreased. Per the stop rule, no lambda sweep, model training, or LCT-CC held-out evaluation was performed.

## Required questions

1. Yes. Frozen B1 mean residual rose from 0.963142 at H0 to 2.800412 at H4 (peaking at 2.869956 at H3), while ground-truth H4 was 0.693765.
2. Both blocks contribute. At observed H4 semantic mean=2.340204 and execution mean=1.400991; semantic is larger, but execution is nontrivial.
3. Yes. Pearson r=0.2420 [0.0838, 0.3909].
4. Yes. Endpoint-error Pearson r=0.3212 [0.1304, 0.5171].
5. Yes. Exactly four iterations reduced the next-cycle residual from 2.939692 to 0.272356.
6. Not under the frozen A5 rule. Mean full RedirectGain remained positive (0.094419), but its clustered lower95 was -0.000694 and endpoint accuracy fell from 0.516260 to 0.401423.
7. No. A1-A4 passed, A5 failed, so M0 was rejected before any optimizer step.
8. None. The development lambda sweep was forbidden after M0 rejection; 0.1, 0.3, and 1.0 were not trained.
9. Not tested; no LCT-CC model exists.
10. Not tested; no LCT-CC model exists.
11. Not tested; no held-out LCT-CC cycle error exists.
12. Not tested; no LCT-CC endpoint result exists.
13. Not tested; no LCT-CC decode/re-encode result exists.
14. Not tested; the prototype comparison was not opened for a nonexistent LCT-CC model.
15. Not tested. Frozen Wave21 B1 remained worse than prototype on continuity; no intervention result exists.
16. Not tested for LCT-CC.
17. Not tested for LCT-CC; the frozen projection itself reduced endpoint identity, warning against equating cycle support with target identity.
18. Not tested across goals for LCT-CC.
19. The exact pair exists (7 cases), but the diagnostic cycle4 map reduced residual while worsening the selected case target distance from 4.650931 to 5.765552; LCT-CC was not trained.
20. C9 is NOT_TESTED because its prerequisite M0 failed.
21. C10 is NOT_TESTED.
22. Wave21 clearly drifts off encoder-decoder-consistent coordinates, and that drift correlates with errors; however, the frozen cycle-supported map is partly misaligned with full language-selected target geometry. Thus pure decoder inconsistency is not sufficient as the primary causal mechanism.
23. Defensible claim: frozen Wave21 language rollout accumulates encoder-decoder cycle drift associated with behavioral/geometric failure, but direct cycle projection trades away statistically reliable full target redirection and target identity.
24. If C9 had passed, the next experiment would be a separately preregistered closed-loop CALVIN rollout comparing frozen B1 and LCT-CC without adding refinement.
25. Because C9 was not reached after M0 failed, next test the language-target/executable-set alignment mechanism directly: characterize goal-specific supported coordinates and preregister a single-factor target-identity alignment model, with the same split and no cycle-based rescue.

## Scientific decision

Cycle drift is a real correlate of Wave21 failure, but the exact frozen correction does not preserve the registered full language effect reliably. Decoder consistency alone is therefore not authorized as the intervention mechanism. The execution-space RedirectGain increased under projection, which is informative but cannot override failed A5 or justify post-hoc training.

## Test-discipline disclosure

Wave21 held-out trajectories were historically opened in Wave21 and were reused because Phase A explicitly requires frozen held-out diagnosis. No Wave22 LCT-CC checkpoint or held-out prediction was ever created. No threshold, seed, lambda, loss, or rescue was changed after the result.
