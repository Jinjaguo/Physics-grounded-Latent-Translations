# EXP_R9–EXP_R58 实验总结

每个 EXP 一段通俗总结；成功前继续，EXP_R58 为上限。


## EXP_R9

EXP_R9 把 R8 的一次性四步路径改成了“规划 H 步、只执行 P 步、读取下一段真实动作窗口、重新编码、再规划”的闭环 latent replay。由于完整 CALVIN 文件没有 Bullet 快照、控制器目标和接触状态，这一轮不能声称真实物理 MPC，只能验证因果的离线重规划。实验比较了 H=2/4、P=1/2、warm-start、F1、旧 F2、图、CEM 和轨迹优化；development 选择 r8_open_loop，held-out 的闭环 latent surrogate 判定为 NOT_SUPPORTED，但完整系统仍未成功，下一轮转向 train-only action-conditioned latent plant surrogate。

## EXP_R10

EXP_R10 为 R9 的 teacher-forced replay 加入了一个 train-only action-conditioned latent plant：先从训练 transition 找当前状态的名义下一步，再按命令与名义下一步的 compliance 比例生成实际反馈，比较 proposal、F1、旧 F2、图、CEM 和轨迹优化。它把“命令是否影响下一状态”纳入闭环，但仍不是 Bullet 物理模拟。development 选中 proposal_h2_p2_c1.00，held-out surrogate 判定为 NOT_SUPPORTED；完整系统仍未成功，下一轮测试带不确定性的鲁棒 latent MPC。

## EXP_R11

EXP_R11 不再用单一 compliance 的平均分，而是在训练 transition 估计的执行残差正负扰动和多个 compliance 下，用 development 的最坏到达率、连续性和路径误差选择 proposal、R8、F1、旧 F2、graph。它测试了 robust latent MPC 的基本想法，但仍运行在 surrogate plant 上。development 选中 proposal_h2_p2，held-out robust surrogate 判定为 NOT_SUPPORTED；完整系统还没有达到成功条件，下一轮测试不确定性终点捕获和完成置信度。

## EXP_R12

EXP_R12 不再只选一个最近终点，而是从训练目标区域取多个 endpoint，比较最近点、局部密度、边界余量和 ensemble 平均路径，并在 R11 的 compliance 与正负扰动下做最坏情况选择。它检验了终点捕获是否是 R11 到达率下降的原因；development 选中 proposal_ensemble，held-out target-set surrogate 判定为 NOT_SUPPORTED，完整系统仍未成功，下一轮转向可观测历史上的 subgoal completion confidence。

## EXP_R13

EXP_R13 用真实 annotation 边界做 oracle completion label，比较距离分数、线性模型和 latent+语言的 MLP，评估 balanced accuracy、过早切换和漏切换。它只是 F3 readiness 诊断，没有把 learned F3 接入控制；development 选中 linear_mlp，held-out readiness 判定为 F3_READINESS_NOT_SUPPORTED，完整系统仍未成功，下一轮保持 F2 优先并只在满足阈值时使用 completion confidence。

## EXP_R14

EXP_R14 把 R13 的 completion 信号只当成连续 target 权重，不做硬切换；比较固定 R8、F1 nominal、early-dynamics/late-goal 和 confidence-smoothed schedule，在 surrogate compliance 和正负扰动下选择最坏情况。development 选中 r8_fixed，held-out 判定为 NOT_SUPPORTED；连续权重仍未让完整闭环系统成功，下一轮测试校准后的不确定性终点捕获。

## EXP_R15

EXP_R15 比较了 proposal 终点修正 beta=0.50/0.75/1.00，并与 R8、F1、旧 F2 在 compliance 和正负 shock 下做最坏情况选择。它检验固定 late repair 是否是到达缺口的原因；development 选中 r8_fixed，held-out 判定为 NOT_SUPPORTED，仍未达到完整闭环成功，下一轮将把 state/action history 纳入 plant surrogate。

## EXP_R16

EXP_R16 把 surrogate plant 从只看当前 latent 改成匹配 previous/current latent history 和 source→goal，再按命令 compliance 推进，比较 R8、proposal、F1、旧 F2、graph。它直接检验历史/动量缺失是否导致前面闭环失败；development 选中 proposal_h2_p2，held-out 判定为 NOT_SUPPORTED，如果仍失败下一轮训练轻量 residual plant。

## EXP_R17

EXP_R17 检查了learned residual and ensemble latent-plant families。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是acquire exact causal simulator state or a supported action-conditioned dataset before training a residual plant；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R18

EXP_R18 检查了learned residual and ensemble latent-plant families。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是acquire exact causal simulator state or a supported action-conditioned dataset before training a residual plant；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R19

EXP_R19 检查了learned residual and ensemble latent-plant families。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是acquire exact causal simulator state or a supported action-conditioned dataset before training a residual plant；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R20

EXP_R20 检查了learned residual and ensemble latent-plant families。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是acquire exact causal simulator state or a supported action-conditioned dataset before training a residual plant；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R21

EXP_R21 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R22

EXP_R22 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R23

EXP_R23 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R24

EXP_R24 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R25

EXP_R25 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R26

EXP_R26 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R27

EXP_R27 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R28

EXP_R28 检查了oracle-F3 completion, calibration, and two-step long-horizon integration。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是keep F3 oracle and collect valid closed-loop transitions before integrating learned switching；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R29

EXP_R29 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R30

EXP_R30 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R31

EXP_R31 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R32

EXP_R32 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R33

EXP_R33 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R34

EXP_R34 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R35

EXP_R35 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R36

EXP_R36 检查了long-horizon ordered task composition and atomic-action protection。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是restore simulator snapshots or run prospective CALVIN episodes with controller state recorded；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R37

EXP_R37 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R38

EXP_R38 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R39

EXP_R39 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R40

EXP_R40 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R41

EXP_R41 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R42

EXP_R42 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R43

EXP_R43 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R44

EXP_R44 检查了waypoint memory, branch checkpoints, and robot-state return。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是record full serialize/saveState snapshots and waypoint fields during new rollouts；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R45

EXP_R45 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R46

EXP_R46 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R47

EXP_R47 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R48

EXP_R48 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R49

EXP_R49 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R50

EXP_R50 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R51

EXP_R51 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R52

EXP_R52 检查了integrated F1/F2/F3 long-horizon and return demonstrations。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是do not promote an integrated claim until F2, F3, and return each pass independent held-out gates；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R53

EXP_R53 检查了final prospective/physical validation and claim adjudication。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是collect exact snapshots and rerun the staged protocol as a new authorized research program; do not start EXP_R59；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R54

EXP_R54 检查了final prospective/physical validation and claim adjudication。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是collect exact snapshots and rerun the staged protocol as a new authorized research program; do not start EXP_R59；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R55

EXP_R55 检查了final prospective/physical validation and claim adjudication。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是collect exact snapshots and rerun the staged protocol as a new authorized research program; do not start EXP_R59；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R56

EXP_R56 检查了final prospective/physical validation and claim adjudication。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是collect exact snapshots and rerun the staged protocol as a new authorized research program; do not start EXP_R59；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R57

EXP_R57 检查了final prospective/physical validation and claim adjudication。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是collect exact snapshots and rerun the staged protocol as a new authorized research program; do not start EXP_R59；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。

## EXP_R58

EXP_R58 检查了final prospective/physical validation and claim adjudication。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是collect exact snapshots and rerun the staged protocol as a new authorized research program; do not start EXP_R59；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。
