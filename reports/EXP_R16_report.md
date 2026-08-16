# EXP_R16 report — history-conditioned latent plant

R16 compared the previous/current history-conditioned plant against the current-only family.

| method | worst arrival | worst decoded diff | worst hidden MSE |
|---|---:|---:|---:|
| r8_fixed | 0.9937 | 1.072888 | 0.9692 |
| proposal_h2_p2 | 0.9832 | 0.807766 | 1.0440 |
| f1_closed_h4_p1 | 0.9371 | 1.143799 | 1.2837 |
| old_f2_closed_h4_p1 | 0.9392 | 1.162259 | 1.2237 |
| graph_mpc_h4_p1 | 0.9518 | 3.040845 | 1.3806 |

Development selected `proposal_h2_p2`. History-plant surrogate claim: **NOT_SUPPORTED**. Overall full-system success remains **false**.
