# EXP_R49 preregistration

## Hypothesis
Three-step ordered composition reveals failure modes hidden by two-step evaluation.

## New method family
`long_horizon` with candidates: replan_each, replan_two, open_loop, graph.

## Frozen components
Representation, decoder, F1, historical F2, and EXP_R8 baseline are frozen.

## Data and split
864 episode-disjoint latent windows: 206 train, 181 development, 477 held-out. Train-only target regions and support are constructed before evaluation.

## Evaluation rule
Select one candidate using development metrics, then open held-out once. Metrics include target arrival, hidden path error, continuity, curvature, support distance, and direction agreement; family-specific F3/prospective metrics are recorded when applicable.

## Success and falsification
A candidate must improve the registered joint gate or pass the family-specific stage threshold. Failure falsifies this mechanism only; it does not erase the language-redirection or action-coordinate claims.
