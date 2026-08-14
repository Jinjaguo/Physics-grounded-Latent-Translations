# Twenty-third wave results: goal-specific executable alignment

Run date: 2026-08-14T18:19:45.248273-04:00

## Outcome

- M1 goal-specific executable alignment: **SUPPORTED_FOR_INTERVENTION**
- Development lambda selection: **NO_CANDIDATE_PASSED**
- C11/C12: **NOT_TESTED**
- Held-out Wave23 test opened: **false**

Phase A supported goal-specific geometry, so all 18 preregistered models were trained. However, no λ achieved either required +0.05 identity improvement. The smallest candidates preserved language redirection, decoded MSE, and continuity, but endpoint and decode/reencode identity worsened. The experiment therefore stopped before held-out inference.

## Required questions

1. Yes, descriptively: all six train-only cores are nonempty (118–243 points), but they are not disjoint; mean pairwise overlap=0.0816.
2. Mean pairwise overlap is 0.0816; maximum 0.1663 for lift_blue_block_slider / lift_red_block_table.
3. Yes. Development Pearson r=0.4919 for distance vs decoded error; distance also tracks poorer target geometry.
4. Yes. Margin/correctness Pearson r=0.7148 and Spearman ρ=0.8442.
5. Yes descriptively. Standardized margin coefficient=0.3317; incremental R²=0.4250.
6. Yes. Global distance improved 3.285295→2.594105 while mean goal-core margin worsened by 0.478635; 72.7% moved to lower margin.
7. Only weakly. Mean execution-margin decrease was 0.004491; the frozen A5 directional definition passed, but CI crossed zero and 49.6% decreased.
8. Yes. M1 passed A1–A5 before training.
9. None. All registered λ values failed development identity-improvement rules.
10. No held-out answer. On development λ=0.03/0.1 retained >=90%; λ=0.3 did not.
11. No held-out answer. All three retained >=90% execution RedirectGain on development.
12. Not tested held-out. Development endpoint macro fell from 0.472422 to 0.465228 for the smallest λ.
13. Not tested held-out. Development decode/reencode fell from 0.408873 to 0.396882 for λ=0.03.
14. Not tested held-out; no GA was selected.
15. Not tested held-out; no GA was selected.
16. Not adjudicated for a selected GA. Development diversity/state-dependence diagnostics are saved, but selection stopped first.
17. Development continuity was no worse for all candidates, but no selected/held-out GA result exists.
18. Not tested held-out; no selected GA exists.
19. Not tested held-out.
20. Held-out was not opened. 6 development cases were analyzed descriptively without cherry-picking.
21. C11 is NOT_TESTED.
22. C12 is NOT_TESTED.
23. Goal-specific geometry explains identity beyond global cycle residual, but the implemented fallback neighborhood lacks source-preceding transition pairing; simply pulling predictions toward execution-nearest core points worsens target margin. Wave21/22 therefore reflect both global/target support conflict and missing transition-conditioned correspondence within each goal core.
24. Defensible claim: train-only goal-core geometry strongly predicts development target identity beyond global cycle consistency, while a preregistered local softmin alignment preserves redirection/continuity but does not repair identity and is not authorized for held-out evaluation.
25. If C11 had passed, the next experiment would be a separately preregistered closed-loop CALVIN comparison of frozen B1 and selected GA from matched simulator states.
26. The language-target-coordinate hypothesis remains supported only as a diagnostic association: goals have structured executable cores and their margins explain identity. What fails is the current alignment intervention, not the earlier causal language-redirection component.

## Scientific conclusion

Goal-specific core geometry is explanatory but the registered alignment operator is not corrective. The likely missing information is transition-conditioned correspondence: train core latents were available, but their source-preceding states were not stored, so the preregistered fallback matched current states in execution space rather than learning a valid path into the goal core.

## Discipline disclosure

Goal cores used train only. M1 and λ selection used development only. No Wave23 held-out inference, post-selection tuning, replacement seed, new λ, cycle loss, classification loss, prototype loss, F2, or DEL was used.
