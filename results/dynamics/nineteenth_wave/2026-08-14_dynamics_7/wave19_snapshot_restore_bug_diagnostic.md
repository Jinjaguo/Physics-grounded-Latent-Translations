# Wave-19 snapshot restore implementation diagnostic

The first contact-rich prospective branch initially failed the frozen zero-error
gate: state max `1.7985e-09`, controller max `7.7884e-09`. The discrepancy was
already present at the restore boundary. The old restore order
`mj_setState -> mj_forward` allowed `mj_forward` to overwrite saved acceleration
and solver warm-start fields.

The corrected order is `mj_setState -> mj_forward -> mj_setState`: the forward
pass rebuilds qpos-dependent geometry for controller queries, and the second
set restores the complete official `mjSTATE_INTEGRATION` payload. Re-running
the frozen formal two-twin certification on the same immutable raw branch then
gave exactly zero state, controller, and object discrepancy over all 210 future
steps, with complete official-predicate agreement. The frozen zero tolerance
was not changed and no representation, F1, or F2 output had been produced.
