# EXP_R7 report — bounded terminal residuals

EXP_R7 compared distributed and late endpoint residual corrections against frozen baselines and the uncorrected proposal.

| method | arrival | decoded first diff | hidden path MSE | F1 cost |
|---|---:|---:|---:|---:|
| linear | 1.0000 | 4.251999 | 1.8414 | 0.8639 |
| f1 | 0.7296 | 1.297798 | 1.8097 | 0.0000 |
| f2 | 0.7715 | 1.264849 | 1.7209 | 0.0218 |
| graph | 1.0000 | 4.656159 | 1.8239 | 0.9372 |
| proposal_base | 0.9623 | 0.907439 | 0.8692 | 0.4003 |
| proposal_repaired | 0.9748 | 0.881743 | 0.8662 | 0.3992 |
| repair_distributed_0.10 | 0.9644 | 0.885536 | 0.8635 | 0.3967 |
| repair_distributed_0.20 | 0.9748 | 0.882553 | 0.8662 | 0.3998 |
| repair_distributed_0.35 | 0.9853 | 0.912459 | 0.8861 | 0.4174 |
| repair_distributed_0.50 | 0.9979 | 0.985616 | 0.9250 | 0.4512 |
| repair_distributed_0.75 | 1.0000 | 1.215286 | 1.0321 | 0.5452 |
| repair_distributed_1.00 | 1.0000 | 1.591875 | 1.1919 | 0.6881 |
| repair_late_0.10 | 0.9644 | 0.890540 | 0.8641 | 0.3960 |
| repair_late_0.20 | 0.9748 | 0.884706 | 0.8649 | 0.3967 |
| repair_late_0.35 | 0.9853 | 0.896430 | 0.8771 | 0.4075 |
| repair_late_0.50 | 0.9979 | 0.933867 | 0.9026 | 0.4298 |
| repair_late_0.75 | 1.0000 | 1.057155 | 0.9745 | 0.4931 |
| repair_late_1.00 | 1.0000 | 1.255994 | 1.0832 | 0.5893 |

Development selected `repair_late_0.35`. EXP_R7 is **NOT_SUPPORTED** (`SUCCESS=False`).
