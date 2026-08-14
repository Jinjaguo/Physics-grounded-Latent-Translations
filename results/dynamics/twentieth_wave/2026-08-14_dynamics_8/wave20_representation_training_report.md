# Wave-20 representation training report

Six preregistered seeds completed exactly 40 epochs for paired R0/R1 and the shuffled semantic control.
R0 and R1 shared initialization seed, train episodes, and epoch batch order. R1 used exactly `2*L_rec + L_sem`.
Fresh confirmation episodes never entered training; EMA epoch 40 was evaluated without epoch search.

| seed | R0 epoch-40 rec | R1 epoch-40 rec | shuffled epoch-40 rec |
|---:|---:|---:|---:|
| 200820 | 0.03879172 | 0.05059035 | 0.05075184 |
| 201820 | 0.04417952 | 0.05317728 | 0.05214967 |
| 202820 | 0.04678866 | 0.05291972 | 0.05274993 |
| 203820 | 0.04568382 | 0.05393532 | 0.05492031 |
| 204820 | 0.04452064 | 0.05472986 | 0.04948607 |
| 205820 | 0.04014108 | 0.05090267 | 0.04793113 |
