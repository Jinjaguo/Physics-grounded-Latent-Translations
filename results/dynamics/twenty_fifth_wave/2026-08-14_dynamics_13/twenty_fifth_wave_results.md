# Twenty-fifth wave results: broad transition implementation sweep

Run date: 2026-08-14T21:16:23.970150-04:00

Models compared: **66**; development eligible: **0**; held-out selected: `[]`.

C15=NOT_TESTED_NO_DEVELOPMENT_CANDIDATE; C16=NOT_TESTED; C17=NOT_TESTED.

## Required questions

1. 257/139/164 transitions; 771/417/492 goal×horizon records.
2. Train-only preferred direction-mode count median=2.0, range 2–3.
3. Log-magnitude BIC selected median K=4.0, the candidate upper bound; therefore no stable discrete magnitude-regime count is identified, only strong heterogeneity.
4. Cancellation ratio/effective-rank Spearman=-0.0423; predictive failure evidence remains descriptive.
5. D2_Wave24
6. Best factorized model: F2_C_separate_heads.
7. M6_K2_mode_residual was the best discrete development model.
8. KNN voting was tested for K=1..4; no vote model qualified.
9. Logistic selectors were evaluated for K=2..4; none qualified.
10. Small MLP selectors were evaluated for K=2..4; none qualified.
11. Mode+residual helped substantially; M6-K2 was best discrete but failed continuity.
12. Best MDN: MDN_K2_argmax; it did not qualify.
13. Soft MoE was slightly better than hard for every K on H4 decoded, but no MoE qualified; best=MoE_K3_soft.
14. Best cVAE: cVAE_D_z2_mean8; it did not qualify.
15. Best flow: Latent_CFM_8step_mean8; Phase-flow passed 5/6 gates but missed endpoint identity.
16. Best diffusion: Latent_Diff_8step_mean8; compact diffusion failed strongly.
17. Best retrieval model: RAT_C_residual; RAT-C passed 5/6 but missed endpoint identity.
18. Causal phase proxies jointly improved a majority of matched families=True; Phase-flow was the strongest new implementation.
19. language_prototype
20. B1_correct_language
21. language_prototype
22. F2_B_D2_direction_learned_magnitude
23. B1_correct_language
24. D4_weighted_affine_a0.001
25. No. O2 discrete-mode oracle H2=1.427315 and H4 decoded=0.064800, both worse than D2.
26. Yes. Best generative oracle=O4_Latent_CFM_16step_mean8_best_of_8, H4 decoded=0.045491.
27. The causal CFM mean closed 90.7% of its H2 D2-to-best-of-8 oracle gap, but still failed the joint eligibility gate.
28. Yes descriptively; current-state dependence is nonzero, but C15 was not tested held-out.
29. Yes on development for B1/CFM/Phase-flow and several compact models under the language-only intervention.
30. Not run: no development candidate qualified, so incremental selected-model retargeting was not authorized.
31. Not run for a selected model; no claim about return or reversibility.
32. All 6 development lift→place cases were compared; Latent_CFM_8step_mean8 jointly improved H2/H4 over D2=False; held-out was unopened.
33. NOT_TESTED_NO_DEVELOPMENT_CANDIDATE
34. NOT_TESTED
35. NOT_TESTED
36. phase-aware latent flow with richer causal history/contact state and explicit identity-continuity mechanism diagnostics
37. Defensible addition: broad development evidence supports continuous phase-aware latent flow as a promising implementation, but no model jointly improved D2 prediction, identity, and continuity; Wave21 causal language redirection remains the central supported claim.
