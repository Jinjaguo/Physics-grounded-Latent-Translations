# Wave 25 execution log

- The first train-only diagnostic stopped because NumPy arrays do not expose `.square()`; the equivalent `singular ** 2` calculation completed.
- The first sweep was interrupted after detecting that D2's third slot used array index 2 instead of the preregistered H4 index 3. The corrected D2 reproduced Wave24 H2=1.208116 and H4 decoded=0.054148 before restarting.
- The next complete sweep reached the oracle stage but stopped because three advanced-index expressions used dictionary length 8 rather than development sample count 139; all three were corrected.
- A reproducibility audit then found model weights were initialized before setting the seed. Training now resets every parameter after seeding; duplicate D5 runs had identical epoch/loss and maximum prediction difference 0. The partial run was discarded.
- The final valid sweep compared 66 candidates and wrote all metrics. No development candidate passed all six gates, so held-out remained unopened.
- The first report attempt treated an oracle audit boolean as a metric row; filtering non-dictionary audit fields completed the report.
