# Wave 29 next experiment

Wave28 selected `BACKBONE_F2_q2` but `READY_FOR_CLOSED_LOOP_RETARGET=NOT_SUPPORTED`. The decisive limitation is that Wave27 prospective transitions do not retain the previous instruction, while Wave21 ordered events are older and have different source/session statistics. Wave29 should collect or reconstruct query-time ordered instruction pairs in the independent source, then test a continuous-time damped q-field and decoder-Jacobian-aware low-rank projection. Keep the action-text VAE, decoder, F1/F2, and F1/F2 primary objectives frozen. Do not append a full future trajectory and do not add an explicit return flag.

Required Wave29 comparisons: static residual vs damped ODE field, learned vs PCA basis, k=2/4/8, F1 vs F2 base, no-switch anchor, h0→h1→h0 cycles, and prospective closed-loop retargeting if the ordered-data gate passes. If ordered data remain unavailable, produce an exact data-collection specification instead of fabricating previous labels.
