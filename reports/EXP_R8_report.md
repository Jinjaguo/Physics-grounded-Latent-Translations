# EXP_R8 report — arrival-first feasible selection

R8 first required development candidates to meet every frozen-baseline bound, then selected the smallest development terminal distance.

| method | arrival | decoded first diff | hidden path MSE | F1 cost |
|---|---:|---:|---:|---:|
| linear | 1.0000 | 4.251999 | 1.8414 | 0.8639 |
| f1 | 0.7296 | 1.297798 | 1.8097 | 0.0000 |
| f2 | 0.7715 | 1.264849 | 1.7209 | 0.0218 |
| graph | 1.0000 | 4.656159 | 1.8239 | 0.9372 |
| proposal_base | 0.9644 | 0.924847 | 0.8760 | 0.4092 |
| proposal_repaired | 0.9727 | 0.879262 | 0.8704 | 0.4035 |
| repair_distributed_0.10 | 0.9644 | 0.891711 | 0.8690 | 0.4030 |
| repair_distributed_0.20 | 0.9727 | 0.879262 | 0.8704 | 0.4035 |
| repair_distributed_0.35 | 0.9811 | 0.897094 | 0.8883 | 0.4173 |
| repair_distributed_0.50 | 1.0000 | 0.959384 | 0.9253 | 0.4474 |
| repair_distributed_0.75 | 1.0000 | 1.171853 | 1.0291 | 0.5353 |
| repair_distributed_1.00 | 1.0000 | 1.530596 | 1.1858 | 0.6725 |
| repair_late_0.10 | 0.9644 | 0.900631 | 0.8700 | 0.4032 |
| repair_late_0.20 | 0.9727 | 0.888759 | 0.8698 | 0.4023 |
| repair_late_0.35 | 0.9811 | 0.893054 | 0.8806 | 0.4107 |
| repair_late_0.50 | 1.0000 | 0.924012 | 0.9047 | 0.4308 |
| repair_late_0.75 | 1.0000 | 1.037526 | 0.9743 | 0.4906 |
| repair_late_1.00 | 1.0000 | 1.227281 | 1.0808 | 0.5840 |

Development feasible candidates: repair_distributed_0.50, repair_late_0.50, repair_late_0.75. Selected `repair_late_0.75`. EXP_R8 is **SUPPORTED** (`SUCCESS=True`).
