# Next experiment after Wave 20

Wave 20 resolved the representation question positively but rejected the frozen offline dynamics gate. Do not
retrain the representation, add seeds, reinterpret the O1 confidence interval, or open the untouched final test.

## Wave-21 single-factor proposal

Test one mechanism suggested directly by the failed O5/O8 geometry endpoints: project each of the existing F2
correction vectors onto the local execution-latent tangent space before applying it.

Freeze before inference:

- the Wave-20 representation checkpoint from seed `202820`;
- the existing Wave-20 semantic, F1, and F2 checkpoints with no optimizer steps;
- four refinement iterations and step size `0.01`;
- 20 nearest neighbors from Wave-19 train execution latents;
- the smallest local-PCA tangent dimension explaining at least 90% variance, exactly matching the already-used
  empirical normal-distance definition;
- no projection coefficient or hyperparameter sweep.

Evaluate `F1`, the frozen original `F2` reference, and `F2_tangent` on the 50 fresh Wave-20 confirmation episodes.
Those episodes were used for representation adjudication but have never been used for dynamics training,
checkpoint selection, or offline dynamics metrics. Gate `F2_tangent` against F1 using the unchanged O1–O8 rules,
and additionally require H8 normal distance no worse than the original F2.

If this prospective gate fails, stop the current refinement family. If it passes, freeze all hashes and open the
untouched Wave-19 final test exactly once for B0–B5 and proposal recovery. Do not use the final test to tune the
tangent projection.
