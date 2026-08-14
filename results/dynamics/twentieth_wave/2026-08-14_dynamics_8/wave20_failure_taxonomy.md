# Wave-20 failure taxonomy

## Infrastructure and representation

- Official LIBERO-10 collection, exact restore, split preservation, and leakage gates passed.
- The prospective representation gate passed with strong bidirectional semantics, motor ratio `1.129209867`, and
  gripper drop `0.000441054`.
- One seed had EMA motor ratio `1.229163475`; this is a reported robustness diagnostic, not a preregistered
  per-seed gate failure.

## Offline dynamics

- O1 failed narrowly: mean ΔAUC(F2−F1) was favorable at `-0.131295617`, but the source-episode clustered 95% CI
  upper bound was `0.001894005`, not strictly below zero.
- O3 failed at H8 by `0.000517911` execution-MSE units (`2.153231758` vs `2.152713848`).
- O5 failed because F2 H8 execution kNN radius was `2.406725152` vs F1 `2.288421908`.
- O8 failed because F2 H4/H8 empirical normal distance was `1.223320502` vs F1 `1.165033846`.
- O2, O4, O6, and O7 passed: refinement improved H4 execution MSE and H8 decoded MSE, and learned corrections
  were positively aligned on average.

The pattern is not a generic divergence or sign failure. Corrections point toward targets often enough to pass the
alignment controls, but their long-horizon execution geometry leaves the empirical train manifold. The frozen
offline gate was therefore rejected.

## Scientific stop

The old 50-episode final test was not opened. B0–B5, direction controls, proposal perturbation recovery, and
closed-loop task success are `NOT_TESTED_OFFLINE_GATE_REJECTION`; they are not negative closed-loop results.
