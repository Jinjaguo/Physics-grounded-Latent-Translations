# EXP_R4 interface audit

EXP_R4 adds only a new trainable multi-step edge proposal. The released representation, decoder, F1 and historical F2 remain frozen. The proposal consumes the current latent and exact target language feature, and emits four latent waypoints; it never receives hidden future actions at evaluation.
