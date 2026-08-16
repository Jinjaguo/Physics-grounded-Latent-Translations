# EXP_R5 report — confidence-gated candidate paths

EXP_R5 selected complete paths with a train-only target-radius, continuity, and support gate.

| method | arrival | decoded first diff | hidden path MSE |
|---|---:|---:|---:|
| linear | 1.0000 | 4.251999 | 1.8414 |
| f1 | 0.7296 | 1.297798 | 1.8097 |
| f2 | 0.7715 | 1.264849 | 1.7209 |
| graph | 1.0000 | 4.656159 | 1.8239 |
| proposal | 0.9602 | 0.929855 | 0.8710 |
| hybrid_select | 1.0000 | 4.643750 | 1.8239 |

Development selected `proposal`. EXP_R5 is **NOT_SUPPORTED** (`SUCCESS=False`).
