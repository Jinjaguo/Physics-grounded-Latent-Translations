# Wave 26 contact/proprioception audit

Exact contact state is absent from the frozen compact CALVIN source. S5 therefore uses only a clearly named causal gripper/motion proxy: current gripper command sign/change plus translation speed, all at or before query time. It is not ground-truth contact.

S7 is `UNAVAILABLE`: compact episode files contain only `rel_actions` and `global_frame_indices`, not TCP velocity, joint velocity, or gripper width state. RGB/future simulator state was not substituted.
