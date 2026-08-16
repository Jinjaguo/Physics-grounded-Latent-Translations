# EXP_R6 report — continuous proposal/linear blends

EXP_R6 selected a fixed convex blend on development data and evaluated that choice once on held-out episodes.

| method | arrival | decoded first diff | hidden path MSE | F1 cost |
|---|---:|---:|---:|---:|
| linear | 1.0000 | 4.251999 | 1.8414 | 0.8639 |
| f1 | 0.7296 | 1.297798 | 1.8097 | 0.0000 |
| f2 | 0.7715 | 1.264849 | 1.7209 | 0.0218 |
| graph | 1.0000 | 4.656159 | 1.8239 | 0.9372 |
| proposal | 0.9727 | 0.887050 | 0.8670 | 0.3895 |
| blend_0.00 | 1.0000 | 4.251999 | 1.8414 | 0.8639 |
| blend_0.25 | 1.0000 | 2.600693 | 1.4262 | 0.6277 |
| blend_0.50 | 1.0000 | 1.512893 | 1.1254 | 0.4692 |
| blend_0.75 | 0.9853 | 0.967794 | 0.9390 | 0.3882 |
| blend_0.90 | 0.9748 | 0.861775 | 0.8821 | 0.3784 |
| blend_1.00 | 0.9727 | 0.887050 | 0.8670 | 0.3895 |

Development selected `proposal`. EXP_R6 is **NOT_SUPPORTED** (`SUCCESS=False`).
