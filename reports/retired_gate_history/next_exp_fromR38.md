# Next experiment from EXP_R38

EXP_R38 established only an interface gate for waypoint memory, branch checkpoints, and robot-state return; it did not open held-out control. The immediate next step is: record full serialize/saveState snapshots and waypoint fields during new rollouts. Keep representation, decoder, F1, old F2, R8 and all prior negative results frozen. EXP_R39 should test a new causal interface or explicitly preserve the gate failure.
