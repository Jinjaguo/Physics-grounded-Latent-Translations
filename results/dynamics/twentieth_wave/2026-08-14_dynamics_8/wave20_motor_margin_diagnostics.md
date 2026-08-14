# Wave-20 motor-margin diagnostics

Translation MSE ratio: `1.129332969`.
Rotation MSE ratio: `1.126310067`.
Per-dimension ratios: `[1.0995509068318674, 1.1867378534761632, 1.1032377002554612, 1.0634845341664285, 1.1162929297505095, 1.153354534277522, 1.0603478956840378]`.

| seed | raw ratio | EMA ratio | EMA improves | below Wave-19 aggregate |
|---:|---:|---:|:---:|:---:|
| 200820 | 1.390063356 | 1.229163475 | True | False |
| 201820 | 1.398186828 | 1.144283760 | True | True |
| 202820 | 1.026062337 | 1.106211902 | False | True |
| 203820 | 1.464611639 | 1.034058928 | True | True |
| 204820 | 1.306999980 | 1.183240298 | True | True |
| 205820 | 1.304462669 | 1.099646038 | True | True |

Train epoch-40 reconstruction, old-development MSE, and fresh-confirmation MSE are preserved in the gate seed rows.
No diagnostic was used to change the frozen gate.
