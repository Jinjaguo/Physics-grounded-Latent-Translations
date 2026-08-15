# Wave 32 — State-conditioned executable low-rank field

Wave31's zero gate collapsed and produced no execution redirect.  Wave32 keeps
the frozen representation/decoder and F1/F2 but uses the causal current latent
to modulate the low-rank basis (C6 state-dependent B).  Training emphasizes
the frozen-decoder action target and uses a smaller continuity weight so the
adapter is not forced to zero.  q={2,4,8} and continuity weights {0.1,0.3,1}
are selected on development, then held-out opens once.

The purpose is to distinguish a globally shared force direction from a local
state-conditioned direction.  No future action or complete future trajectory
is an input, and no explicit return flag is added.
