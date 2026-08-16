# EXP_R10 interface audit

R10 keeps the exact EXP_R9 latent/action interface and adds a train-only action-conditioned surrogate plant. Each surrogate step uses only a nearest train nominal transition and the current commanded latent; the held-out future is used only for evaluation. This is not an exact Bullet simulator because retained episodes lack snapshots, contacts, and controller targets.
