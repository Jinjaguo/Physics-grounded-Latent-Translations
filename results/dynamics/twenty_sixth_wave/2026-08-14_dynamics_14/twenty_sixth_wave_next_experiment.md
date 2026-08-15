# Twenty-sixth wave next experiment

## Evidence-based decision

Wave26 labels: MIXED_EVIDENCE, DATA_LIMITED, IDENTITY_CONTINUITY_TRADEOFF_PERSISTS. C18=NOT_TESTED, C19=NOT_SUPPORTED, C20=NOT_SUPPORTED, C21=MIXED, C22=MIXED, and `READY_FOR_RETARGETING_TEST=False`.

The next experiment should **collect genuinely new source-session-disjoint paired transitions with synchronized gripper/TCP/contact state, then rerun RAT-C and retrieval-initialized flow controls**. Keep Wave21's same-state causal language redirection central and retain the incremental interface: recent causal history + current action coordinate + next atomic language → one local executable transition. Do not reopen DEL, global cycle projection, or static endpoint attraction.

## Concrete Wave27 design

If readiness is false, acquire genuinely new source-session-disjoint paired transitions with synchronized gripper width, TCP/joint velocity and explicit contact signals, then train a compact recurrent phase encoder feeding the best Wave26 flow family (`Flow_S0_Prior-CFM`). Match it against `State_S0_RAT-C` and keep one frozen held-out evaluation. If readiness is true, use `[]` for a simulator matched-state retarget/interruption/return-to-stored-waypoint pilot; call return recoverable-state return, not physical time reversal.

## Research relation

[CoLA-Flow](https://arxiv.org/abs/2601.23087) motivates temporally coherent continuous latent-action flow; [LAFM](https://arxiv.org/abs/2606.23420) motivates an adaptive library of state-selected source priors; [3D FlowMatch Actor](https://arxiv.org/abs/2508.11002) motivates low-latency targeted flow architectures; and [BAKU](https://arxiv.org/abs/2406.07539) motivates retaining matched MLP/MoE/VQ action-head controls. A newer [Guided Action Flow](https://arxiv.org/abs/2607.02092) result motivates a TRAIN-only transition critic for causal sample selection, but only after collecting success/failure or transition-quality labels. These are implementation hypotheses; Wave26's own held-out evidence determines the branch.
