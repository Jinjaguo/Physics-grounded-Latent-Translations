# Wave 27 final report questions



1. Available routes were the official full human-play archive and the too-small debug archive; policy, scripted, and local teleoperation routes were unavailable.

2. 52 genuinely new authoritative source sessions, disjoint from Wave21 sessions 0–30.

3. 407 certified, non-overlapping 128-frame paired transitions.

4. {"lift_blue_block_slider": 78, "lift_red_block_table": 83, "place_in_slider": 79, "push_pink_block_right": 54, "turn_off_lightbulb": 56, "turn_on_lightbulb": 57}

5. 0%; the archive has no true contact signal.

6. 407/407 records have measured gripper width/state.

7. 0 measured TCP-velocity records; causal finite differences are labeled derived.

8. 0 measured joint-velocity records; causal finite differences are labeled derived.

9. Yes—split and bootstrap use authoritative source-session IDs; within-session transition ranges do not overlap.

10. Yes: the independent library grew to 407 transitions across 52 sessions; retrieval-distance diagnostics quantify expansion.

11. LN25 improved over L0 in the matched-method mean.

12. LN50 improved further in the matched-method mean.

13. LN100 improved further in the matched-method mean.

14. NEW-only's best matched model was `Scale_NEW-only_F2-C` on independent development sessions.

15. The nested same-collector curve diagnoses sample/session count; L0 versus NEW-only also changes source domain, so those effects are not conflated.

16. Synchronized physical state beat PH0; selected `PH1`.

17. The matched sweep selected `PH1`; the manifest gives exact gripper/joint/TCP/scene fields.

18. NOT_TESTED because true contact is absent; proxies were not relabeled.

19. PH4/PH5 explicitly model causal phase; `PH1` was selected on development only.

20. Memory K=4/8/16/32 was swept; the best K is empirical rather than assumed monotonic.

21. State-aware retrieval helped relative to the goal mean.

22. The best learned/hybrid retrieval cell was `Retrieval_R5_learned_scorer_K8`.

23. Candidate scoring used TRAIN library outcomes plus causal query state; query futures were never inputs.

24. RIF did not beat RAT-C on the development composite.

25. The best retrieval/state flow did not beat global-source CFM.

26. Prior-CFM was evaluated under all PH conditions; the overall selected physical state was `PH1`.

27. Streaming-CFM's exact matched result is in `wave27_streaming_cfm_results.md`.

28. Temporal flow did not jointly improve identity and continuity relative to global flow.

29. Heteroscedastic heads were tested; their reported mean-prediction result determines whether uncertainty helped.

30. Causal best-of-samples selection was tested against retrieval support; no ground-truth oracle selected samples.

31. Yes. Transition gradients were nonzero, frozen representation gradients were zero, and no cycle projection was used.

32. Decoder supervision helped H4 decoded error without worsening continuity.

33. Transition contrast used full H1/H2/H4 paths plus wrong-language dynamic negatives; matched results are reported.

34. The development Pareto front has 35 models: Control_PH1_Phys-F2C, Control_PH1_RAT-C, Core_L0_PH0_RAT-C, Core_LN100_PH0_Phys-F2C, Core_LN100_PH0_RAT-C, Core_LN100_PH0_TC-CFM, Core_LN25_PH0_RAT-C, Core_LN25_PH0_TC-CFM, Core_LN50_PH0_Phys-F2C, Core_LN50_PH0_RAT-C, Core_LN50_PH0_TC-CFM, Core_NEW-only_PH0_Phys-F2C, Core_NEW-only_PH0_RAT-C, Core_NEW-only_PH0_TC-CFM, Core_NEW-only_PH1_RIF, Core_NEW-only_PH1_TC-CFM, Flow_PH1_MultiCandidate-CFM, Flow_PH1_RIF-A_random, Flow_PH1_RIF-C_initialize, Flow_PH1_Temporal-CFM, Objective_PH1_decoded, Physical_PH0_F2-C, Physical_PH0_RIF-C, Physical_PH1_RIF-C, Physical_PH2_Prior-CFM, Physical_PH2_RIF-C, Physical_PH3_RIF-C, Physical_PH4_F2-C, Physical_PH4_RIF-C, Physical_PH5_RIF-C, Scale_LN100_F2-C, Scale_LN100_Prior-CFM, Scale_LN100_RIF-C, Scale_LN50_F2-C, Scale_NEW-only_Prior-CFM.

35. Frozen candidates: Core_LN100_PH0_RAT-C, Flow_PH1_RIF-A_random, Core_LN100_PH0_Phys-F2C, Scale_LN100_Prior-CFM.

36. Prospective winner: `Core_LN100_PH0_RAT-C`.

37. C23: SUPPORTED.

38. C24: SUPPORTED.

39. C25: SUPPORTED.

40. C26: NOT_SUPPORTED.

41. C27: NOT_SUPPORTED.

42. C28: SUPPORTED.

43. C29: SUPPORTED.

44. READY_FOR_RETARGETING_TEST is False.

45. Lift→place chain improvement is NOT_TESTED because success-certified chains were not collected.

46. Contact-phase heterogeneity is NOT_TESTED because true contact is absent.

47. Cross-collector generalization is NOT_TESTED; only official-human-play source-session generalization is tested.

48. The limiting factor is determined by the failed readiness criteria together with the data and physical ablations; unavailable signals are not guessed.

49. target the failed readiness criteria with phase-balanced independent collection and compact physical retrieval

50. Defensible claim: changing only next-goal language causally redirects a session-independent local latent trajectory to the extent shown by prospective cluster-bootstrap CIs; retrieval and phase are named only when their claims are supported.
