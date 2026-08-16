# EXP_R12 report — target-set terminal capture

R12 compared train-only endpoint policies under compliance and execution shocks.

| policy | worst arrival | worst decoded diff | worst hidden MSE |
|---|---:|---:|---:|
| r8_nearest | 0.9895 | 1.074058 | 0.9664 |
| proposal_density | 0.9832 | 0.779813 | 0.9790 |
| proposal_margin | 0.9853 | 0.786339 | 0.9757 |
| proposal_ensemble | 0.9832 | 0.779018 | 0.9787 |
| f1_closed_h4_p1 | 0.9832 | 1.157366 | 1.1388 |
| old_f2_closed_h4_p1 | 0.9769 | 1.172860 | 1.1143 |
| graph_mpc_h4_p1 | 0.9644 | 2.540853 | 1.2140 |

Development selected `proposal_ensemble`. Target-set surrogate claim: **NOT_SUPPORTED**. Overall full-system success remains **false**.
