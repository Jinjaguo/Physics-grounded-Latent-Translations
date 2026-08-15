# Wave 26 execution log

- The first sweep stopped before training because Python 3.8 does not implement dictionary-union runtime semantics. The feature-manifest merge and six spec updates were changed to explicit `update`; no held-out data was touched.
- The second sweep stopped at the frozen Wave25 anchor check: the generic checkpoint loader used `seed+steps`, while the registered Phase-flow sweep used `seed+71`. The loader now restores the original phase-family sampling rule. Exact reproduction then reached H2=0.833001912 and H4 decoded=0.044799913 (maximum drift <=1e-7).
- The final valid sweep ran 79 development entries. S7 was unavailable because compact sources lack robot/TCP/joint state; D3 was unavailable because all 31 independent sessions are already assigned. Neither was imputed.
- Exact contact was absent; S5 is explicitly a causal gripper/motion proxy, not ground-truth contact.
- The `Objective_decoded` implementation duplicated the multi-horizon latent path proxy rather than applying a differentiable frozen-decoder loss. It is retained in the exhaustive record but excluded from objective/mechanism claims. Frozen-decoder metrics were still computed for every candidate. A true decoded-trajectory loss must be preregistered in a later development wave; it was not added after held-out was opened.
- Candidate selection froze Prior-CFM, History-CFM, and RAT-C before any test array materialization. Held-out was opened once for those checkpoints only; no winner tuning followed.
- The first report draft used an overly permissive existence check for C19. It was corrected to require the same flow to beat a matched control on both held-out H2 full and H4 decoded MSE. C19 is therefore NOT_SUPPORTED; models/results were unchanged.
