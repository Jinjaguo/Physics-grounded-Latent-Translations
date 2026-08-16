# Next experiment from EXP_R16

R16 changed only the surrogate plant to previous/current history matching. It selected `proposal_h2_p2` with claim NOT_SUPPORTED. If unsupported, EXP_R17 should test a small learned residual plant with train-only sequence augmentation; if supported, repeat the oracle-F3 closed-loop gate on the history plant.
