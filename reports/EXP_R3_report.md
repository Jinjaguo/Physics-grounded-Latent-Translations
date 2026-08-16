# EXP_R3 report — complete-episode transition paths

EXP_R3 replaced isolated boundary windows with complete official CALVIN episode annotations. It used four H16 chunks before and four hidden chunks after each valid boundary, with episode-disjoint train/development/held-out splits.

| method | arrival | decoded first diff | hidden path MSE | F1 cost |
|---|---:|---:|---:|---:|
| linear | 1.0000 | 4.251999 | 1.8414 | 0.8639 |
| f1 | 0.7296 | 1.297798 | 1.8097 | 0.0000 |
| f2 | 0.7715 | 1.264849 | 1.7209 | 0.0218 |
| retrieved_delta | 0.7820 | 5.860597 | 2.4189 | 1.9956 |
| graph | 1.0000 | 4.656159 | 1.8239 | 0.9372 |
| traj_full | 1.0000 | 3.445999 | 1.8027 | 0.8412 |

Development selected `f2`. EXP_R3 is **NOT_SUPPORTED** (`SUCCESS=False`). F3 remains oracle and no return or closed-loop MPC claim is made.
