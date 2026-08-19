# EXP_F1–F4 Snapshot Model-State 勘误

EXP_F5审计发现：F1–F4 legacy `LiberoSnapshot`保存了MuJoCo integration state、controller和observable，但没有保存LIBERO placement sampler写入的`model.body_pos/body_quat`。不同seed fresh env后，integration state可以exact-zero，fixed-body placement仍可相差最多1.76 cm，并导致rendered observation不同。

因此，F1–F4中“同一实验、同一failure下所有choices从相同restored env开始”的matched对照仍成立，实际interventions、step traces和相对method outcomes仍是有效证据；但“fresh env完整等于最初failure-generation world”与逐像素replay的表述不成立，旧audit没有覆盖model-level state。

F5起snapshot schema已加入`model.body_pos/body_quat`。跨不同seed env修复验证为：integration、model arrays、agentview、wrist、EEF、gripper最大误差均为0。F6只使用修复后的full-state snapshots重建canonical benchmark；legacy snapshots不作为最终confirmation依据。
