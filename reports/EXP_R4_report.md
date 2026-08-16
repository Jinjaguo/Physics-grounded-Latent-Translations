# EXP_R4 report — learned local edge proposals

EXP_R4 trained a small four-step proposal on train complete-episode transitions and compared single proposals, multi-hypothesis selection, and terminal repair with frozen baselines.

| method | arrival | decoded first diff | hidden path MSE | F1 cost |
|---|---:|---:|---:|---:|
| linear | 1.0000 | 4.251999 | 1.8414 | 0.8639 |
| f1 | 0.7296 | 1.297798 | 1.8097 | 0.0000 |
| f2 | 0.7715 | 1.264849 | 1.7209 | 0.0218 |
| graph | 1.0000 | 4.656159 | 1.8239 | 0.9372 |
| edge_proposal | 0.9539 | 0.894549 | 0.8759 | 0.3969 |
| multi_hypothesis | 0.9769 | 0.865163 | 0.8633 | 0.3975 |
| edge_proposal_repaired | 0.9706 | 0.851615 | 0.8705 | 0.3921 |

Development selected `edge_proposal_repaired`. EXP_R4 is **NOT_SUPPORTED** (`SUCCESS=False`). The proposal remains offline; no F3 learning, closed-loop MPC, or return claim is made.
