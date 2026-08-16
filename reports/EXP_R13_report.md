# EXP_R13 report — oracle-boundary F3 readiness

This was a readiness diagnostic, not F3 integration.

| method | held-out AUROC | balanced accuracy | early switch | late miss |
|---|---:|---:|---:|---:|
| distance_score | 0.7368 | 0.6874 | 0.2558 | 0.3695 |
| linear_mlp | 0.7852 | 0.6929 | 0.0823 | 0.5320 |
| mlp_fusion | 0.7697 | 0.6059 | 0.0204 | 0.7678 |

Development selected `linear_mlp`. F3 readiness: **F3_READINESS_NOT_SUPPORTED**. F3 was not promoted because closed-loop F2 remains unsupported.
