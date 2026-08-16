# EXP_R17–EXP_R67 scientific reboot execution log

This log records the reboot as a method-driven sequence. The pre-reboot
gate-only reports remain copied under `reports/retired_gate_history/` and are
not counted here.

| range | focus | outcome |
|---|---|---|
| R17 | horizon-dependent repair schedules | linear schedule improved continuity but did not pass the joint gate |
| R18–R30 | goal/history/multimodal transition, ensemble uncertainty, shooting, CEM/MPPI, graph, terminal-set, adaptive horizon, trust region | R19/R20/R21/R28 passed offline stage gates; the rest remained unsupported |
| R31–R40 | tangent, support, value, distillation, retrieval+optimization, multiresolution, tube/risk/transfer/authority | R32/R33/R40 passed offline stage gates; no physical claim |
| R41–R46 | completion, hazard/change-point, fusion, confidence-gated authority | F3 thresholds were not met; R46 was rerun after replacing a shared fallback with distinct gates |
| R47–R55 | two/three-step composition, learned F3, retarget, interruption, latent/waypoint/checkpoint return | several offline stage gates passed; return remains a latent/waypoint surrogate |
| R56–R59 | F1/F2/F3 integration and module ablations | module-specific differences were evaluated; no full-system success |
| R60–R62 | matched counterfactual prefixes and action-conditioned surrogate benchmark | R60 was rerun after separating matched/random/goal-swap implementations; causal benchmark remains offline |
| R63–R67 | distributional sampler, transfer, stress, prospective collection, final adjudication | R67 hard stop reached without overall success |

All experiments used 864 episode-disjoint windows (206 train, 181
development, 477 held-out), selected on development, and opened held-out once.
The one post-wave disk audit reported about 911.8 GB available, above the
200 GB floor. No R68 was started.
