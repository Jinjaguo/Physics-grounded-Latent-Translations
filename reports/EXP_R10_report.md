# EXP_R10 report — action-conditioned latent plant surrogate

R10 replaced R9 teacher-forced observation with a train-only nominal-transition plant and explicit command compliance. It remains an offline surrogate, not physical MPC.

| method | actual arrival | decoded first diff | actual hidden MSE | tracking MSE | replans |
|---|---:|---:|---:|---:|---:|
| proposal_h2_p2_c1.00 | 0.9958 | 0.721392 | 0.9656 | 0.0000 | 2.0 |
| proposal_h4_p2_c1.00 | 0.9958 | 0.721392 | 0.9656 | 0.0000 | 2.0 |
| proposal_h2_p1_c1.00 | 0.9832 | 0.690950 | 0.9950 | 0.0000 | 4.0 |
| proposal_h4_p1_c1.00 | 0.9832 | 0.690950 | 0.9950 | 0.0000 | 4.0 |
| proposal_h2_p2_c0.75 | 0.9832 | 0.716896 | 0.9167 | 0.0618 | 2.0 |
| proposal_h4_p2_c0.75 | 0.9832 | 0.716896 | 0.9167 | 0.0618 | 2.0 |
| proposal_h2_p1_c0.75 | 0.9769 | 0.657496 | 0.9255 | 0.0587 | 4.0 |
| proposal_h4_p1_c0.75 | 0.9769 | 0.657496 | 0.9255 | 0.0587 | 4.0 |
| proposal_h2_p1_c0.50 | 1.0000 | 0.692048 | 0.9495 | 0.2293 | 4.0 |
| proposal_h4_p1_c0.50 | 1.0000 | 0.692048 | 0.9495 | 0.2293 | 4.0 |
| proposal_h2_p2_c0.50 | 1.0000 | 0.739569 | 0.9484 | 0.2451 | 2.0 |
| proposal_h4_p2_c0.50 | 1.0000 | 0.739569 | 0.9484 | 0.2451 | 2.0 |
| r8_open_loop_c1.00 | 1.0000 | 1.035578 | 0.9800 | 0.0000 | 1.0 |
| r8_open_loop_c0.75 | 1.0000 | 1.035578 | 0.9075 | 0.0664 | 1.0 |
| proposal_h2_p1_c0.25 | 1.0000 | 0.809460 | 1.1215 | 0.5229 | 4.0 |
| proposal_h4_p1_c0.25 | 1.0000 | 0.809460 | 1.1215 | 0.5229 | 4.0 |
| proposal_h2_p2_c0.25 | 1.0000 | 0.789781 | 1.1109 | 0.5532 | 2.0 |
| proposal_h4_p2_c0.25 | 1.0000 | 0.789781 | 1.1109 | 0.5532 | 2.0 |
| old_f2_closed_h4_p1_c0.75 | 0.9644 | 1.070933 | 1.1801 | 0.0817 | 4.0 |
| f1_closed_h4_p1_c0.75 | 0.9644 | 1.076038 | 1.2091 | 0.0842 | 4.0 |

Development selected `proposal_h2_p2_c1.00`. CEM and trajectory candidates were registered but invalidated before held-out because their repeated surrogate budget exceeded CPU time; no held-out values were read. Action-conditioned surrogate claim: **NOT_SUPPORTED**. Overall full-system success remains **false**: no physical/exact simulator feedback, learned F3, long-horizon sequencing, or return.
