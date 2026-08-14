# Wave-19 development snapshot certification

The simulator-only development gate used 100 restore-and-replay trials across all 10 official
`libero_10` tasks. No π0.5, representation, F1, or F2 output was used.

- median segment maximum discrepancy: `0`
- P95 segment maximum discrepancy: `0`
- maximum discrepancy: `0`
- official terminal-predicate agreement: `1.000`
- all values finite: `True`
- gate: `PASS`

The frozen certification tolerances are empirical simulator-determinism tolerances and cannot be changed after
source-policy or F1/F2 outputs are observed.
