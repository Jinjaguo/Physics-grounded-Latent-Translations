# Wave-19 predicate-order certification diagnostic

`task01_attempt025` initially stopped collection even though integration, controller, and object discrepancies
were all exactly zero. A replay A/B showed zero predicate mismatches when the official predicate was read in the
same order as source collection, but one transient mismatch at step 252 when certification read it after the
canonicalizing `mj_forward` inside snapshot capture.

The collector and certifier now share the frozen order `step -> check_success -> canonical snapshot -> physical
diagnostics`. The immutable raw episode then recertified with state/controller maximum error `0.0`, so no tolerance,
trajectory, action, or success label was changed.
