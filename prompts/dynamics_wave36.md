# Wave36 decoder-Jacobian action transport

Wave35 showed that a temporal/state bridge can predict a direction but does not preserve action continuity.  Wave36 therefore predicts a small action-space displacement from the ordered event and current/past state, then transports that displacement through the frozen decoder's local Jacobian into a latent force.  The VAE, decoder, F1, and F2 remain frozen.

Run several methods together: Jacobian transpose, damped pseudoinverse, execution-only projection, phase-gated transport, and task-balanced/cycle-symmetric objectives.  Sweep q=2/4/6, PCA/random action bases, and multiple damping/continuity weights.  No future action or latent is an inference input.  Continue to Wave37 unless the success gate passes; only success or completion of Wave78 may end the program.
