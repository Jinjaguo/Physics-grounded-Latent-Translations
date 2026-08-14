# Wave-19 invalid Numba-mode recovery diagnostic

One recovery command incorrectly set `NUMBA_DISABLE_JIT=1` to work around a cache issue seen only in restricted
test launches. Formal raw collection used robosuite with JIT enabled. The mismatched controller execution path
produced state/controller maxima of approximately `1127` / `48.3`, so that recovery result is invalid rather than
evidence against the formal snapshot.

With the original JIT mode restored, the same immutable raw episode passed at exact `0.0` state and controller
error. The experiment continues only under the original execution mode.
