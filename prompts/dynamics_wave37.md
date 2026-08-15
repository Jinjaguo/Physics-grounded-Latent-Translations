# Wave37 cycle-consistent task-balanced force bridge

Wave36 showed that decoder-Jacobian transport alone is insufficient.  Wave37
tests a causal bridge whose force is antisymmetric under swapping the ordered
current and arriving instructions.  The training objective includes forward
target matching, a reverse-cycle loss, a no-switch zero-force anchor, and an
optional task-balanced weight.  At inference it receives only the current
state/history and the current/arriving instruction pair.

Compare pair-only, state-pair, and phase-pair bridges, q=2/4/8, PCA/random
low-rank bases, and cycle/no-switch weights.  Keep VAE/decoder/F1/F2 frozen.
Failure requires Wave38; only success or completion of Wave78 ends the program.
