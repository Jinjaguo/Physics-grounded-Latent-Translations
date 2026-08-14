# Wave-19 statistical report

The completed representation analysis used 10,000 source-episode-clustered, task-stratified bootstrap
replicates with seed `190819`. Correct-language semantic delta over the stronger of shuffled-language and
reconstruction-only was `0.940000` action-to-text (95% interval `[0.916667, 0.960000]`) and `0.900000`
text-to-action (95% interval `[0.866667, 0.933333]`).

The motor condition failed: continuous MSE ratio was `1.200444393` versus the frozen maximum `1.2`; the absolute
excess over the threshold was `0.000000759028`. Gripper sign accuracy passed (`0.987560` correct-language versus
`0.989337` reconstruction-only, well inside the allowed `0.05` drop).

No offline dynamics or closed-loop statistic exists because the mandatory stop condition prevented those phases.
The 50 held-out test episodes were not read.
