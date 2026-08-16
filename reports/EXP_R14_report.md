# EXP_R14 report — progress-gated target authority

R14 used completion-distance only as a continuous F2 blend, not a hard F3 switch.

| method | worst arrival | worst decoded diff | worst hidden MSE |
|---|---:|---:|---:|
| r8_fixed | 0.9727 | 0.742565 | 0.9895 |
| f1_nominal | 0.9182 | 1.157366 | 1.3187 |
| early_dyn_late_goal | 0.9329 | 1.053027 | 1.2355 |
| confidence_smoothed | 0.9329 | 1.073062 | 1.2542 |

Development selected `r8_fixed`. Progress-gated surrogate claim: **NOT_SUPPORTED**. Overall full-system success remains **false**.
