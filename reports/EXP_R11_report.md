# EXP_R11 report — robust latent MPC surrogate

R11 selected candidates by worst-case development arrival, continuity, and hidden-path error across train-derived positive/negative execution shocks and compliance conditions.

| method | worst arrival | worst decoded diff | worst hidden MSE |
|---|---:|---:|---:|
| r8_open_loop | 0.9895 | 1.081429 | 1.1269 |
| proposal_h2_p2 | 0.9811 | 0.857073 | 1.1380 |
| f1_closed_h4_p1 | 0.9832 | 1.157366 | 1.1388 |
| old_f2_closed_h4_p1 | 0.9769 | 1.172860 | 1.1143 |
| graph_mpc_h4_p1 | 0.9644 | 2.540853 | 1.2140 |

Development selected `proposal_h2_p2`. Robust surrogate claim: **NOT_SUPPORTED**. Overall full-system success remains **false** because physical/exact feedback, learned F3, long-horizon sequencing, and return remain unavailable.
