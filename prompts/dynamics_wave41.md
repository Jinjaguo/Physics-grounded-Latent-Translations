# Wave41 calibrated local trust-region force

Wave40's semantic/execution split improved redirection but remained too large
and poorly calibrated in action space.  Wave41 predicts both a force direction
and a confidence/radius, applying a small local trust-region update.  Compare
fixed radius, state-adaptive radius, and two-head uncertainty calibration with
q=2/4/8 and PCA/random bases.  Keep the frozen VAE/decoder/F1/F2 and use no
future inputs.  Continue unless success or Wave78.
