# Next experiment from EXP_R10

EXP_R10 tested a train-only action-conditioned latent plant with compliance values [0.25, 0.5, 0.75, 1.0]. It selected `proposal_h2_p2_c1.00` and established only a surrogate result (NOT_SUPPORTED); physical closed-loop feedback remains unavailable. EXP_R11 should test disturbance-robust latent MPC: train-only residual uncertainty sets, robust terminal capture, and proposal/F1/F2/graph baselines under fixed perturbation budgets. Keep representation, decoder, F1, F2, R8 planner, and the R10 plant frozen.
