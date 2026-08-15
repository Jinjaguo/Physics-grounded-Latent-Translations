# Twenty-sixth wave results: rich causal state × structured continuous flow

Run date: 2026-08-14T22:24:14.580795-04:00

Development models=79; held-out candidates=3; held-out Pareto=['Flow_S0_History-CFM', 'Flow_S0_Prior-CFM', 'State_S0_RAT-C'].

## Claim outcomes

```json
{
  "C18_rich_causal_state_matters": "NOT_TESTED",
  "C19_continuous_flow_strongest_family": "NOT_SUPPORTED",
  "C20_enriched_flow_reduces_identity_continuity_tradeoff": "NOT_SUPPORTED",
  "C21_more_transition_data_helps": "MIXED",
  "C22_language_and_state_shape_transition_distribution": "MIXED",
  "READY_FOR_RETARGETING_TEST": false,
  "ready_models": [],
  "best_state_configuration": "S0",
  "best_flow_family": "Flow_S0_Prior-CFM",
  "best_nonflow_control": "State_S0_RAT-C",
  "best_data_condition": "D2",
  "state_limitation_evidence": 0,
  "data_limitation_evidence": [
    true,
    true,
    true
  ],
  "model_limitation_evidence": true,
  "representation_limitation_evidence": false,
  "language_redirect_preserved": true,
  "execution_redirect_preserved": false,
  "identity_improved": false,
  "decode_reencode_improved": false,
  "continuity_improved": true,
  "outcome_labels": [
    "MIXED_EVIDENCE",
    "DATA_LIMITED",
    "IDENTITY_CONTINUITY_TRADEOFF_PERSISTS"
  ],
  "recommended_wave27_direction": "collect genuinely new source-session-disjoint paired transitions with synchronized gripper/TCP/contact state, then rerun RAT-C and retrieval-initialized flow controls"
}
```

## Required questions

1. S1 vs S0 Phase-CFM H2: 0.979817 vs 0.904921.
2. S2 Phase-CFM H2=1.035142; incremental benefit is reported, not assumed.
3. S3 action history H2=1.091402.
4. S4 gripper state H2=1.035058.
5. S5 causal contact proxy H2=1.050287; exact contact was unavailable.
6. S6 learned PCA phase state H2=1.079869.
7. S7 minimal proprioception was unavailable in the frozen compact source and was not imputed.
8. Best causal state by matched development score: S0.
9. Simultaneous held-out endpoint/continuity improvement over Wave25 Phase_flow: [].
10. Best History/phase flow candidate: Flow_S0_Prior-CFM.
11. Prior-CFM was compared for every selected state; see flow-family table.
12. R-CFM was compared with leave-one-out TRAIN retrieval initialization.
13. Streaming-CFM used scaled previous displacement as causal source initialization.
14. TC-CFM jointly modeled H1/H2/H4 with data-relative temporal losses.
15. Hetero-CFM learned state-dependent diagonal source scale.
16. MP-CFM used three learned continuous source branches with a causal gate.
17. Multi-horizon supervision is isolated in the objective table.
18. Transition-contrastive supervision is isolated with wrong-row transition negatives.
19. Exact frozen-decoder trajectory supervision was not completed: the registered Objective_decoded run duplicated the multi-horizon path proxy, so it is excluded from mechanism claims; all candidates were still evaluated through the frozen decoder.
20. Adaptive continuity used a TRAIN-batch P90 displacement-velocity threshold.
21. Causal retrieval-support selection was run with N=8 and no future ground truth.
22. Matched F2-C state results are in Table B.
23. Matched RAT-C state results are in Table B.
24. VQ-Transition K=8 and K=16 were trained as learned discrete controls.
25. Development Pareto frontier contains 10 models.
26. Frozen held-out candidates: ['Flow_S0_Prior-CFM', 'Flow_S0_History-CFM', 'State_S0_RAT-C'].
27. D0->D1->D2 monotonic H2 flags: [True, True, True].
28. D3 was unavailable because all independent compact sessions were already assigned.
29. No D3 performance claim was made.
30. Primary taxonomy: ['MIXED_EVIDENCE', 'DATA_LIMITED', 'IDENTITY_CONTINUITY_TRADEOFF_PERSISTS'].
31. C18=NOT_TESTED.
32. C19=NOT_SUPPORTED.
33. C20=NOT_SUPPORTED.
34. C21=MIXED.
35. C22=MIXED.
36. READY_FOR_RETARGETING_TEST=False.
37. Held-out inference latency range=0.029--0.519 ms/query.
38. Lift->place cases=6; simultaneous held-out global identity/continuity improvement=False.
39. Wave27 implementation direction: collect genuinely new source-session-disjoint paired transitions with synchronized gripper/TCP/contact state, then rerun RAT-C and retrieval-initialized flow controls.
40. Defensible paper claim: language and causal history both alter the learned local transition distribution when the registered held-out C22 criterion is supported; otherwise Wave21 causal language redirection remains the central result and Wave26 is comparative implementation evidence.
