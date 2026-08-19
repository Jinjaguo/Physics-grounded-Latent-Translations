# EXP_G1–EXP_G40 闭环因果实验总报告

本文按实际执行顺序记录 G 系列的 40 个有效实验。每一节只描述真正生成过数据、训练过模型或执行过 simulator/controller 干预的内容；预运行 gate、smoke、失败后修复目录和单纯报告工作均不另算 EXP。所有结论以各 EXP 保存的 rollout、checkpoint、dataset、metric 和独立 audit 为准。

## 逐实验记录

### EXP_G1

G1 的假设是完整保存 MuJoCo、robosuite 环境、控制器和观测状态后，可以从同一物理边界精确恢复并公平比较不同动作。实验在 10 个 LIBERO 任务各取一个开发集 checkpoint，对 source、damped、reverse、hold 四类 16 步控制做了 60 条真实 rollout、960 个控制步，并对 source 重复执行。相同动作的 integration state 和 latent 最大差异都是 0，不同动作的末态都超过重复误差，证明 `checkpoint -> proposal -> executed action -> realized state -> re-encode` 原语可用；但这里的 latent 仍主要编码动作历史，也没有测试任务完成。这个结果消除了接口 gate，使 G2 可以真正比较 F1/F2 提案。

### EXP_G2

G2 假设旧 F1/F2 产生的 latent 提案只有在真实执行、恢复和重规划后才可能有价值，因此在 10 个开发 episode 上建立四步 receding matched-state shooting，并比较 source oracle、open-loop/receding F1/F2、causal state 和 causal latent，共执行 1,120 个提交步和 1,760 个候选步。16 步局部误差上 causal latent 最好，为 0.0220，优于 open-loop F1/F2 的 0.0319/0.0331 和 causal state 的 0.0496；40 次决策中 17 次选了 learned/copy latent 提案。不过所有方法任务成功率都是 0，因为 horizon 太短，且 F1 单独通常优于旧 F2。G3 因而把同一问题扩展到完整剩余任务，而不把局部误差优势当成功。

### EXP_G3

G3 假设 G2 的局部 causal-latent 优势能累积成完整原子任务成功，于是在 10 个任务的晚期 checkpoint 上运行 128–236 步完整 horizon，比较 source、receding F1/F2、causal state 和 causal latent。source 成功 5/10，所有可部署方法均为 0/10；causal state 虽有唯一的正平均进展和最低 jerk，causal latent 仍因四步 oracle 路径评分过于短视而漂移。该结果推翻“局部目标误差即可代表长时操控”的假设，并暴露 action-only latent 缺少接触/阶段信息，促使 G4 改为每个真实状态都重新预测动作的 state-conditioned controller。

### EXP_G4

G4 假设物理状态和对象状态能补足 G3 的阶段缺失，并测试 nearest-state retrieval、state-only MLP、state+rolling-action-latent MLP。所有模型只用训练轨迹，部署后每一步都读取 simulator 实际状态，不再 teacher force。结果 retrieval 为 0/10，state MLP 为 1/10，matched state+latent MLP 为 2/10，相比 receding F1 的 0/10 首次出现非 oracle 完成；latent 也在同容量比较中带来 1 个成功，但两种 MLP jerk 高、平均误差仍差。G5 因而检验真正的时序记忆是否能保留这点 latent 信息并稳定接触控制。

### EXP_G5

G5 假设 GRU 记忆完整动作/状态历史会比 G4 的单步 MLP 更适合接触阶段，因此为 10 个任务各训练 state-only 和 state+latent GRU，共 20 个模型，并在同一 10 个 checkpoint 上从实际状态 recurrent 更新。两种 GRU 都是 0/10，尽管 state+latent GRU 把失败末态误差降到 0.272、state GRU 的动作更平滑，仍不如 G4 的 2/10 完成率。结论是缺少时序记忆并非唯一瓶颈，真正问题是 expert-only 训练无法纠正 learner 进入的新状态；G6 转向用真实 learner rollout 收集纠正数据。

### EXP_G6

G6 假设 train-only DAgger 式纠正聚合能缓解分布偏移。实验先让冻结的 G4 state+latent 策略从每任务两个训练 checkpoint 运行，得到 20 条真实 rollout 和 2,560 个 learner-visited 状态，再训练 equal-weight 与 correction×3 两组 task-local MLP 并回到同一开发集。equal aggregation 仅 1/10，三倍纠正为 2/10，最多追平 G4，且后者平均末态误差恶化到 0.528。原因是按相同时间索引配对的 expert action 在已经偏离的状态上并不一定正确；G7 因此尝试短动作块和多模态输出，而不是继续调样本权重。

### EXP_G7

G7 假设一次预测 16 步动作块或用四模态高斯混合能保留接触序列并减少逐步抖动，共训练 30 个 task-local 模型：state chunk、state+latent chunk、latent mixture。每执行四步后都从真实状态重规划。state chunk 2/10、latent chunk 1/10、latent mixture 0/10，均未超过 G4；latent 模型虽降低失败误差但没有完成增益，state chunk 还因未约束 gripper logit 产生极高 raw jerk。多模态和 latent 假设均失败，G8 改问缺失信息是否来自视觉对象/接触观测。

### EXP_G8

G8 假设双相机视觉能突破纯状态策略 2/10 的上限，并在 140 条训练 episode、7,402 个 policy-issue window 上训练共享 task-conditioned CNN，比较 visual-state 与等容量 visual-state+action-latent。两者在 10 个晚期 checkpoint 都是 0/10；纯视觉误差 0.385，latent 版本虽更平滑却把误差扩大到 0.658，低 imitation loss 没有转化为 rollout recovery。G8 因而否定轻量视觉模仿与 latent 增益，G9 转而接入官方 π0.5 大模型作为可靠 F1，并只让 causal selector 决定候选。

### EXP_G9

G9 假设强的 observation/language-conditioned π0.5 F1 加上真实执行后的 state-value 选择可以突破旧控制器上限。实验加载官方 14 GB `pi05_libero`，从 140 个训练 episode 训练每任务 state-only 和 state+latent progress value，在 10 个任务上比较单样本、open latent ranking、causal state value、causal state+latent value。单样本 5/10，causal state 达 6/10，超过 G4 的 2/10 和 archived source 的 5/10；但加入 action latent 反而降至 2/10，open latent 也只有 4/10。由此确定 π0.5 是可用 F1、state causal selection 是较强基线，而 latent 不应主导；G10 开始真正的 `lift -> place` 组合。

### EXP_G10

G10 假设硬 current-action gating 加 oracle 4 cm×3-step F3 能保护 lift，并从实际 lifted state 切换到 place，比较 future-visible full prompt、hard gate realized retarget、restart place 和 place-only。5 个 task-5 开发初态中所有方法都 lift 5/5，但 realized retarget place 0/5，future-visible 与 place-only各 1/5；restart 同为 0/5，却把 final error 从 0.0816 恶化到 0.5023 并丢失 lift 5/5。严格 gating 没显示 lift 优势，且 place prompt 本身会隐式完成 pickup，说明 π0.5 语义并非干净原子化；但 current-state 继续明显优于初态重启。G11 因而专门修复 post-lift place 控制。

### EXP_G11

G11 假设明确告诉策略“物体已经 lifted”的上下文能修复 G10 的 place 阶段，同时检验 post-lift suffix state/latent value。14 个训练 episode 生成 1,646 个 post-lift 状态并训练 matched value，从 5 个精确 realized lifted state 真实执行。contextual-place full-value 成功 4/5，优于 place-only 和原 full prompt 的各 3/5；suffix-state 与 suffix-state+latent 仍都是 3/5，但 error 由 0.0561 降到 0.0482 和 0.0468，latent 只提供很弱的误差 tie-break 而无成功增益。place 瓶颈基本解除后，G12 把关键问题转到 autonomous F3。

### EXP_G12

G12 假设训练集时序窗口能自主检测 lift 完成，并检验 rolling action latent 是否改善 F3。14 个训练 episode 构造 562 个 issue window，训练 state-window GRU 与 matched state+latent GRU，再与 oracle、train-median fixed、future-visible 比较真实五初态组合。state F3 成功 4/5，匹配 oracle 4/5，优于 fixed 3/5 和 future-visible 1/5，5/5 切换且无 premature；state+latent 只有 2/5，虽训练分类完美却控制泛化失败。G12 首次证明 autonomous switch、current-state retarget 和 causal F2 可以组成闭环两动作系统，但 latent 对 F3 是负贡献；G13 转向冻结测试和 latent-support F2。

### EXP_G13

G13 假设训练动作 latent 的 support distance 能约束候选并在 frozen test 保留开发优势。14 个训练 episode 提供 pre/post-lift latent support，开发集选择 λ=0.05 时为 4/5、λ=0 时 2/5；但冻结测试完全反转，latent support 2/5，而 latent-disabled、single-proposal 和 F3-disabled future-visible 都是 3/5，restart 1/5、open loop 0/5，teacher replay 5/5。完整系统虽优于 restart/open loop，却没有击败简单基线，说明 latent 几何过拟合开发集。G14 因而重新训练由物理 outcome 监督的 action representation，而不沿用旧几何。

### EXP_G14

G14 假设用 phase、未来 EEF/book displacement 和 next action 联合监督的 16-D outcome latent 会改善 F3。14 个训练 episode 产生 2,784 个动作历史窗口，并在此前完全未出现的官方 init 40–49 上执行 outcome-latent、old latent、state、fixed 与 future-visible。五种方法全是 10/10，outcome latent 仅凭 switch-error 8.5 步优于 old 9.5、state 10.5、fixed 11.5，但 state 的 final error 更低。这个 cohort 太容易，不能证明 latent 必要；G15 因而预注册真实对象位移扰动和完整模块消融。

### EXP_G15

G15 假设 outcome-latent F3 加两候选 causal F2 在物理分布移位下会优于 state/no-F2/no-F3/restart。最初混合对角扰动使 attempt46 物理爆炸，保留为 gate 后改成经验证的 4 cm cardinal shift，在 10 个初态执行 70 条正式 rollout。full、state F3、future-visible、restart 都 10/10，single/no-F2 9/10，open loop 0/10、unperturbed replay 2/10；full 只比 no-F2 多一例，完全未优于 state/no-F3/restart。因而闭环 feedback 强烈有效，outcome latent 和模块分解却未被支持；G16 把 latent 移到能直接生成动作的 F2 坐标。

### EXP_G16

G16 假设 phase-conditioned 4-D PCA/AE/VAE action-chunk 坐标可围绕 π0.5 生成可执行候选。14 条训练演示产生 2,658 个 chunk，开发执行选择 PCA 后，在 10 个 6 cm 扰动初态比较 latent shooting、raw π0.5 shooting、raw noise、decoded base、single、open loop、replay。latent 仅 2/10，decoded base 1/10，而 raw shooting 和 single 都 10/10；open loop/replay 0/10，raw Gaussian search 1/10。把已经 observation/language-conditioned 的 π0.5 动作投影进无条件 demonstration manifold 丢掉关键分量，是主要破坏源。G17 因而要求 identity-safe residual：候选零永远保持原生动作。

### EXP_G17

G17 假设只生成 state-conditioned residual、保留 raw candidate0，能避免 G16 的重构伤害。G15/G16 的 2,340 个真实分支训练 conditional AE 和八码 residual codebook；正式 retry 保留可独立重解码上下文。在 7 cm 扰动上 ranked residual latent 10/10，优于 shuffled 8/10、single 9/10、outcome-head-only 7/10、open loop 0/10、replay 1/10，说明 residual 排名和真实 branch feedback 都有信号；但 equal-budget raw π0.5 shooting 同为 10/10，且 error/jerk 更好。latent 仍不必要，G18 转而保留所有原生动作字节，只用 representation 预测 realized transition 来选候选。

### EXP_G18

G18 假设由 5,423 条候选级真实转移学习的 contrastive transition latent 能比 direct/state ranker 更好选原生 proposal。经过两个无效大位移 cohort 后，正式使用接触稳定的 7.5 cm x-axis 扰动；latent 8/10，direct transition MLP 9/10，state value 9/10，first candidate 和 single 各 8/10，open loop/replay 0/10。latent switch error 最小但成功率更低，说明压缩 realized effect 没带来选择优势；闭环 F2 仍明显优于无反馈。G19 因而把 latent 改做 checkpoint/waypoint memory，并要求真正执行 return controller。

### EXP_G19

G19 假设 sequence latent 能选出更有利 checkpoint，物理返回后再从 realized recovered state 继续。2,110 个序列窗口训练 contrastive、autoencoder、direct memory；每个测试 case 保存四个完整 checkpoint，再施加 3.5–4.1 cm OSC disturbance，执行比例 OSC return。所有 30 条物理 return 都达到 EEF/book tolerance，证明接口成功，但 latent 与 direct return 都 8/10，latest 和 restart 6/10，no-return continuation 反而 9/10 且 error 最低，simulator restore upper bound 9/10。物理接近旧 checkpoint 并不保证更好的 policy continuation distribution；G20 因而只在确有收益时触发 return。

### EXP_G20

G20 假设 leave-one-attempt-out latent gate 能从 G19 的 paired outcome 判断何时值得 return。实验训练 latent、direct-state、joint logistic gate，并从同一 disturbed snapshot 执行可选 OSC return 后 fresh π0.5 continuation。latent gate 8/10、joint 8/10、never 8/10、always 8/10，direct 7/10，saved-outcome oracle 9/10；latent endpoint error 还差于大多数基线。因为 learned gate 没严格优于固定策略，checkpoint return 从主系统移除。G21 将 F2 资源问题改写为“何时值得执行多个 matched candidate branch”。

### EXP_G21

G21 假设由真实 candidate-zero regret 监督的 compact latent 能只在必要时开启三候选 shooting，从而兼顾成功和计算。实验从 G15–G18 重建 1,859 个 pre-decision group，训练 contrastive latent、autoencoding latent 和 direct MLP，并在新 rollout 中比较 latent/direct trigger、always shooting、single 和 matched random trigger。latent 7/10、direct 与 always 各 8/10，single 与 random 各 9/10；latent 虽比 always 少 53% branch cost，仍丢失成功，最简单 single 又最好又最便宜。G22 因而冻结 single-proposal F2，把 representation 转到由下游 counterfactual outcome 监督的 F3。

### EXP_G22

G22 假设“现在切换、再等 5 步、再等 10 步”的真实下游 regret 比高度边界更适合训练 F3。实验在 20 个 checkpoint 各执行三条真实 continuation，得到 60 个反事实 rollout；三种 delay 的成功都是 18–19/20，只有一组改变 binary success，监督对 timing 有连续差异但失败对比很弱。训练的 counterfactual latent F3 在正式测试 8/10、switch error 21.2，direct regret 7/10，而旧 state-window F3 为 10/10、error 6.0；fixed 8/10、future-visible 7/10。latent 压缩没有改善状态 F3，G23 因而再次测试 representation 能否作为 jointly trained executable action bottleneck。

### EXP_G23

G23 假设把 action bottleneck 与 realized transition、utility、success 联合训练能使 decoded latent MPC 安全。7,188 条候选转移、2,642 个 group 训练 VQ、8-D continuous、action-only AE 和 direct world model，共 40 个 held-attempt checkpoint。正式结果 continuous latent 4/10、VQ 0/10、action-only 2/10、decoded-first 5/10，而 direct world model 与 physical shooting 各 8/10、single raw π0.5 9/10。即使 joint training，解码也平均改动原生控制 0.32，产生 reconstruction/model-exploitation error；G24 因而彻底停止改动作字节，改为观察条件的 visual transition prediction。

### EXP_G24

G24 假设把 action coordinate 与真实双视角视觉变化对齐，能够在不解码动作的情况下优于直接模型。实验先发现并修复 legacy snapshot 缺失 OSC derived fields 的真实问题，再从 10 个 canonical 初态收集 1,242 条物理候选转移、414 个 matched group，训练 aligned/unaligned visual latent、direct visual/state。正式结果 aligned、unaligned、direct visual 都 9/10，direct state 7/10、shuffled 6/10、physical shooting 与 single 8/10；aligned 没有严格胜出，direct visual error 还最低。支持的是“保留 native action 的视觉预测”，不是 latent bottleneck；G25 继续检验 causal transition history 是否能给 latent 真正的时序作用。

### EXP_G25

G25 假设 aligned latent-history GRU 能利用过去已提交的真实 transition 来改进下一候选排名。1,242 条 branch 被组织成 414 个因果时序 group，并训练等参数 aligned-latent GRU 与 raw-action/effect GRU；未提交候选 outcome 从不进入在线 history。aligned 7/10，raw-history 9/10，memoryless aligned 9/10；shuffled 或每步 reset latent 都仍是 7/10。结果说明 conventional raw history 可以有用，但 aligned recurrent state 既未提高成功也未通过 shuffled/reset 机制检验。G26 因而问 latent ensemble uncertainty 是否比单点预测更适合控制风险。

### EXP_G26

G26 假设 aligned ensemble 的 candidate-specific disagreement 可识别不可靠 proposal。实验采用 group bootstrap 和嵌套校准，为 aligned/direct visual 各训练 100 个校准与 100 个最终 checkpoint，再执行 90 条新 rollout，并把 uncertainty 故意错配到别的候选做机制消融。aligned uncertainty 与 direct 都 8/10，mean-only 7/10，physical/single 各 9/10，错误归属的 shuffled disagreement 反而 10/10。bootstrap averaging 可能有益，但正确 uncertainty 不是成功原因；G27 因而消除候选 index/order 混淆，训练严格 permutation-equivariant set ranker。

### EXP_G27

G27 假设 aligned-coordinate DeepSets 能利用 proposal 集合的相对结构，并通过六种 permutation 验证等变性。10 个 held-attempt fold 在 414 个三候选 group 上训练 aligned set、等参数 raw set、aligned point；离线 aligned set accuracy 最高 0.7685，且恢复排序误差低于 1e-6。可是 100 条 prospective rollout 中 aligned set 8/10，raw set 与 aligned point 各 9/10，physical 9/10、single 7/10；集合上下文降低而非提高控制。G28 因为此前不同方法的 π0.5 随机提案不同，增加 explicit common random numbers 来排除运气解释。

### EXP_G28

G28 假设在完全 matched 的 π0.5 flow noise 下 aligned representation 的优势会显现。服务器扩展为每个请求接收显式 10×32 noise，并验证相同 observation/noise 逐 bit 复现；30 个 held-attempt point model 和 set model 在 3×10×8=240 条 rollout 上比较。aligned point 23/30，unaligned point 27/30，single 27/30；aligned/raw set 都 26/30，physical 24/30，open loop 0/30。aligned 离线 ranking accuracy 0.7475 虽最高，却有 0 比 4 的 paired 劣势于 single，明确排除 proposal-noise luck。G29 将表征目标改成 realized physical action effect，并用 conservative gating。

### EXP_G29

G29 假设八维 action-effect coordinate 加 conservative baseline bootstrapping 能只在高置信时干预。10 个 fold 为 effect-coordinate、raw-effect、direct 三类各训练三成员 ensemble，共 90 模型；240 条新 rollout 中 coordinate 真实选择过 650 次非零候选。结果 conservative coordinate 23/30，raw-effect/direct 各 24/30，always-rank effect 24/30，G28 aligned 与 physical 各 25/30，single 22/30。coordinate 确实偶尔修复 single 失败，但未击败强基线，且 conservative gate 也没优于 always-rank；G30 因而检验这个 coordinate 是否至少能作为 CEM 的生成搜索空间。

### EXP_G30

G30 假设由 realized effect 监督的 8-D action coordinate 可让 CEM 生成原生集合外的优质控制。60 个 latent/raw ensemble 经过 held-out 选择后，每次做 32 sample×3 iteration 搜索并执行生成前缀。latent CEM 0/30，latent random 1/30，raw CEM 6/30，单纯 encode/decode base 19/30，而原生 single 26/30、aligned native point 与 physical 各 25/30；latent CEM 几乎每次都选生成动作且 jerk 超过原生三倍。模型优化系统性利用离线 effect/value error，证明该 latent 不是安全生成控制空间。G31 回到只执行 native proposal，并让 verifier 在试执行后决定 accept/restore/retry。

### EXP_G31

G31 假设 action/effect latent verifier 可根据真实试执行结果做自适应 checkpoint recovery。40 个 held-attempt verifier 比较 latent、raw visual、physical 和 shuffled latent；每个候选都从完整 checkpoint 执行，接受则保留 realized state，拒绝则精确恢复。latent 22/30，raw/physical/shuffled 都 25/30，exhaustive shooting 26/30；physical 只用 shooting 35.8% 的 rejected steps 保留 25/26 个成功，而 latent 对各强 verifier 都是负 paired balance。自适应 recovery 机制成立，latent correspondence 不成立；G32 因而移除 representation，用透明 realized utility stopping rule。

### EXP_G32

G32 假设不经过 latent 的 stage-specific measured-effect rule 能以较少 branch cost 达到或超过 exhaustive shooting。用 G24 的 414 个 matched group 在同一 EXP 内选择 absolute、improvement、stage-specific threshold，再做 240 条新 rollout。stage-specific 27/30、rejected 9,720，优于 learned physical 与 exhaustive 的各 24/30；improvement 26/30、只 rejected 2,420；但 aligned point 也 27/30 且零试执行，single 26/30。透明 causal stopping 满足本轮规则，却不是全局最佳，也没有 latent 贡献。G33 随后加入真正的在线物体扰动，测试谁能检测并恢复机制变化。

### EXP_G33

G33 假设 aligned action-latent residual 能作为 disturbance memory。每条 rollout 在决策 8 和 24 对 candidate0 真实平移 book 3 cm，比较 aligned residual、raw visual residual、direct physical residual、utility、oracle、always、no recovery 和 clean，共 240 条。latent 只检出 30/60、成功 14/30；raw visual 39/60、15/30；direct physical 检出 60/60、成功 24/30，十个 paired win、零 loss 于 latent；oracle/always 23/30，no recovery 4/30，clean 23/30。它强力证明 F2 的 `execute -> observe -> restore/retry` 能恢复扰动，但作用来自 raw-action-conditioned physical effect，而非 latent visual residual。G34 转向 F3 的 monotonic phase coordinate。

### EXP_G34

G34 假设严格单调的 action-phase coordinate 能改善 autonomous switching。14 个训练 episode 产生 562 个 issue-step，训练 42 个 fold checkpoint 和 3 个 final model：monotonic coordinate、direct binary、shuffled temporal order；再固定 G33 physical recovery 执行 240 条 rollout。monotonic 23/30、switch error 12.0，direct 24/30、state 25/30、future-visible 26/30，故意打乱时序的 coordinate 却 27/30、error 49.2。即使 physical oracle error 仅 2 步也只有 24/30，证明边界精确度不是该任务成功代理，future information 也未必有害。G35 因而扩展到明确的 `grasp -> lift -> place` 三阶段和两个 switch，提高识别力。

### EXP_G35

G35 假设一个共享 language-conditioned action coordinate 能在连续三阶段中自主判断两次完成并保护当前动作。先做 12 条 prompt intervention，选出能稳定 pre-lift grasp 的指令；再收集 14 条训练轨迹、576 个 issue-step，训练 coordinate、direct binary、no-language coordinate、state 四类 56 个 fold checkpoint，并在 240 条带两次 3 cm 扰动的 rollout 中比较八机制。shared coordinate 25/30，state/fixed 各 26/30，no-language 与 physical oracle 各 27/30；shared 在 grasp/lift 都 30/30 premature。最重要的是移除 F2 后从 25/30 崩到 7/30、lift retention 29 降到 17。三阶段系统可工作，F2 必要，但 language/action coordinate 对 F3 有害；G36 将 latent 再移到 physical-effect verifier。

### EXP_G36

G36 假设由真实六维 EEF/book displacement 对齐的 physical-effect coordinate 会比 direct residual 更好检测扰动。1,242 条 G24 branch 训练 aligned、predictive bottleneck、shuffled 共 30 个 held-attempt checkpoint，并固定 G35 state F3，在 240 条 rollout 中执行五候选和两次 book shift。aligned 检出 60/60 但 strict 23/30、false trigger 21.66%、rejected 1,895；direct 同样 60/60，却 24/30、8.53%、880；no recovery 仅 4/30。latent 含有真实 pairing 信息但压缩后校准更差，direct physical F2 仍是赢家。G37 因而检验 language-conditioned effect latent 是否至少能跨任务、小样本迁移。

### EXP_G37

G37 假设共享 language-conditioned action/effect latent 能从任务 0、3、8 的少量 target row 迁移扰动检测。实验在每任务六个 train checkpoint 自动选物体，收集 270 条新 causal branch（216 clean、54 disturbed），训练 shared latent、pooled/task direct、no-language、shuffled 共 42 模型，并在 0/12/36 adaptation budget 比较；正式使用 12-row budget 执行 84 条 rollout。shared、pooled direct、no-language、shuffled 都 6/12，task direct 与 oracle 5/12，no recovery 4/12；shared false trigger 高达 72.8%，task8 连 oracle 都 0/4。多轮 repair 最终保存每步官方 predicate 并通过审计。language 没有提供迁移信号，G38 改用 latent 只排序被 direct verifier 判定后的 fallback proposal。

### EXP_G38

G38 假设从 G36 的 13,330 个真实 trial 学到的 action-support latent 能在 candidate0 被拒后，更好排列候选 1–4，而不再承担 anomaly detection。40 个 held-attempt 模型比较 support latent、pairwise latent、direct classifier、shuffled；210 条三阶段 rollout 中 main latent 24/30，direct/native 23/30，shuffled 22/30，pairwise 25/30，raw diversity 24/30，physical oracle 28/30。main 对 direct/native/shuffled 有小幅正 paired balance，说明 support label 有信号，但 rejected 1,240 高于 native 1,205，且输给 pairwise，未满足 success+efficiency 规则。固定集合限制仍可能掩盖生成价值，因此 G39 最后测试新的 corrective proposal。

### EXP_G39

G39 假设 conditional VAE action latent 能从“candidate0 失败、后续候选成功”的 401 个 causal pair 生成一个原生五候选之外的 corrective 五步前缀，并胜过 direct regressor、raw interpolation 与 native search。30 个 held-attempt generator checkpoint 后执行 210 条新 rollout：latent strict 25/30、direct 27/30、native 25/30、interpolation 25/30、pairwise 24/30、shuffled 22/30、oracle 26/30；latent 对 direct 为 1 win/3 loss，虽把 rejected steps 从 native 的 1,820 降到 1,010，成功规则仍失败。审计还发现生成前缀距最近 native 的中位 RMS 0.0614，但只有 58.6% 高于 0.05、最小 0.00693，在线常塌回原生邻域。由此停止增加 latent F2 变体，G40 进行最终 direct/state 简化与全部关键消融。

### EXP_G40

G40 的最终假设分两层：简化系统（π0.5 native proposal 作为 F1、direct physical-residual branch recovery 作为 F2、state completion 作为 F3）应在 30 个 matched 扰动 case 上达到至少 24 次严格成功并击败关键消融；现有 latent intervention 还必须严格超过它，才支持 Actions-as-Coordinates。九路、270 条正式 rollout 的结果是 full direct/state 28/30，与 oracle F3 完全相同；latent 26/30，paired 为 full 3 胜/1 负；teacher-forced 26/30，restart 与 future-visible unprotected 各 21/30，no-F2 和无 checkpoint recovery 各 4/30。no-F3 按要求两次切换的 strict 指标为 0/30，却获得 29/30 官方 task success，这是最关键的反例：显式 executive 保证程序语义，但 π0.5 单靠 grasp prompt 已经常完成整个任务。审计重算了 270 条链、12,412 次决策、14,500 次恢复和 540 次扰动并通过。简化闭环控制成功，latent 贡献与“F3 是任务成功所必需”均不成立，因此在用户指定的 G40 终点停止。

## 当前最佳系统

当前最可靠、证据最完整的系统不是原 proposal 中的三套 learned latent 模块，而是一个明显更简单的组合：F1 直接使用官方 π0.5 根据当前双相机图像、proprioception 和当前 prompt 产生五个 native 10×7 action chunk；F2 对每个五步前缀做真实 simulator branch execution，用 held-attempt direct model 预测 EEF/book 六维位移并比较 realized residual，接受时直接沿 reached state 继续，拒绝时精确恢复 simulator/controller checkpoint 后试下一候选；F3 使用 G35 的 state/action-history binary completion model 切换 `grasp -> lift -> place`。G40 中它在两个真实 book displacement 下达到 28/30 strict success，与 physical oracle F3 完全相同，并显著优于无 F2、无恢复、重启和 future-unprotected 对照。这套系统可以被复现和审计，但应称为“direct physical-feedback closed-loop controller”，不能称为 action-latent coordinate controller。

## 真正证明了什么

第一，因果闭环和 realized feedback 是确定有效的。G15–G18 一再显示 initial-observation open loop/replay 接近 0，而 receding π0.5 能到 8–10/10；G33、G35、G36、G40 更直接地把外部扰动、执行动作、realized next state、restore/retry 和最终成功连成完整链。G40 中移除 F2 后 strict success 从 28/30 降到 4/30，final error 从 0.0568 增到 0.4501；只检测失败却不给 checkpoint recovery authority 也同为 4/30。这证明 F2 的作用不是一个离线 score，而是执行后真正改变下一条控制轨迹的 branch recovery。

第二，从当前物理状态继续下一动作比回到 episode 初态更合理。G10 已显示 realized retarget error 远小于 restart；G40 又在完整扰动 protocol 中得到 full 28/30、restart 21/30，full 有七个 paired win、没有 loss，且 restart 的 lift retention 仅 14/30。因而“切换后从实际 reached state 重新观测和规划”是受支持的系统原则。

第三，autonomous state F3 足以替代 oracle 来执行显式程序。G12 的 state-window F3 在开发集匹配 oracle，G40 又在 30 个 case 上与 oracle 的 strict outcome 逐 case 完全一致。future-visible full prompt 的 strict success 为 21/30、缺少五次完整 switch，说明 prompt protection 对程序合规有帮助；teacher-forced progress 为 26/30，也略低于 state F3。这些结果支持“可以自主切换并记录动作程序”，但不等于 F3 提高普通 task reward。

第四，40 轮结果持续否定“learned action latent 提供不可替代的控制价值”。少数早期或局部证据存在，例如 G2 的 16 步局部误差、G4 的 2/10 对 1/10、G17 的 ranked-vs-shuffled residual、G38 的小幅 paired ranking signal；但一旦加入同预算 raw/direct baseline、长 horizon、frozen test、common random numbers 或完整 integrated ablation，latent 都未保持优势。G40 是最终判定：latent 26/30，direct 28/30，且 direct 有 3–1 paired 优势。因此最终目标中“没有 latent 就不能复现系统增益”的条件没有满足。

## F1 可行性评估

F1 作为“根据当前 observation、state 和当前动作语义产生局部控制”的功能是可行的，π0.5 已经很好地承担了它；G9 从旧 controller 的 2/10 上限提高到 6/10，后续 task-5 扰动实验中 native proposal 常达到 8–10/10。但原设想中的“learned action-progress latent F1”没有被证明。旧 F1/F2 latent 在 G2 只有短 horizon 优势，G3 长 horizon 全部失败；G16/G23/G30 表明 decode 或优化 latent action 会破坏 π0.5 已编码的视觉、语言和实例信息；G39 direct correction 又优于 latent correction。更深的结构问题是 π0.5 本身并不遵守干净的原子边界：place prompt 会先 pickup，grasp prompt 在 G40 甚至得到 29/30 完整任务成功。因此 F1 的功能可行，但把它解释成一个外部、可分离、由 action coordinate 推进的原子模块，在当前 policy/benchmark 上不可识别。

## F2 可行性评估

F2 是三者中证据最强、也最接近真正可行的模块。direct action-conditioned physical residual 能在 G33/G36 检出 60/60 人工扰动，并在 G40 将 strict success 从无恢复的 4/30 提到 28/30；exact checkpoint restoration、proposal execution 和 accepted-state continuation 都有 machine-verifiable trace。不可行之处主要不是闭环思想，而是当前实现方式的外推范围。它依赖 simulator 能把已经执行过的失败分支精确“擦除”，真实机器人无法无代价回到同一世界状态；它还依赖已知 book body、精确 EEF/object pose 和任务特定位移模型。G37 又显示接触 predicate 甚至可能因一次 MuJoCo forward/capture 而变化。因而这套 F2 在 LIBERO simulator 内成立，迁移到真实机器人必须用安全可逆探测、可信 world model 或实际 recovery motion 替代魔法式 rollback。至于 latent F2，G16、G17、G18、G23–G33、G36–G39 的一致结果说明当前数据量、监督目标和 bottleneck 会丢失 native proposal 的实例特异信息或被 optimizer 利用；这是本项目方案的实证不可行原因，但不是对所有可能 latent MPC 的数学不可能性证明。

## F3 可行性评估

F3 作为显式程序计数器和可解释 switch mechanism 是可行的：state F3 能做两次 autonomous switch，并在 G40 与 oracle strict outcome 完全一致。可是把 F3 写成“完成任务所必需的 learned action-coordinate executive”在当前 benchmark 上不可行。最直接证据是 no-F3 只有 0/30 strict switch compliance，却有 29/30 official success，高于 full 的 28/30；strict 指标把“没有 switch”定义为失败，因此 full 对 no-F3 的 strict 优势部分是协议定义，而不是 reward 改善。G34 的 shuffled-order coordinate 以 49 步 boundary error 得到最高 27/30，G35 的 no-language coordinate 又胜过 language coordinate，future-visible prompt 多次表现很强。这说明 completion timing、language identity 和 task success 在这里没有被唯一识别。若要证明 F3 的科学必要性，需要未来动作与当前动作真正冲突、premature authority 会稳定导致失败、且低层 policy 不会自动补做所有前置动作的新 benchmark。

## F1/F2/F3 整体方案为何可能不可行

整体分解并非在工程上完全不能运行——G40 已证明它能运行，而且 direct 版本很强。真正不可行的是原论文想要的那条独特因果解释。第一，低层 π0.5 已内部完成了感知、语义、前置动作和长时规划，外部 F1/F3 边界与模型内部能力重叠；第二，F2 需要的 task-relevant effect 在此任务中只有 EEF/book 六维量，直接模型信息更完整，latent bottleneck 没有压缩优势；第三，训练数据主要来自单一 task-5 或很少几个任务，language 在任务内几乎恒定，无法识别 language-conditioned coordinate，G37 的跨任务小样本也没有补足；第四，offline reconstruction、ranking、uncertainty、phase accuracy 与 closed-loop reward 长期脱钩，representation 很容易优化代理指标而不是 policy improvement；第五，π0.5 随机 proposal 的质量方差很大，common-noise G28 以后才排除了许多候选运气；第六，exact simulator rollback 让 F2 强大，却使整套机制不直接对应不可复位的真实物理系统。这些不是“代码还没调好”的单点问题，而是 representation identifiability、模块职责重叠、benchmark 可识别性和部署接口四个层面的共同限制。

## 被推翻的原始设想与仍未完成的目标

被推翻的核心设想包括：旧 F2 latent 能累积局部优势；action history latent 稳定改善 controller；visual/action alignment、uncertainty、set context、effect bottleneck 或 CEM 会自然带来 causal control gain；checkpoint proximity 是好的 recovery objective；language-conditioned coordinate 会优于 no-language；精确 phase boundary 会提高 task success；latent generator 能胜过 direct generator。仍成立的较小结论是：强 F1/native proposal、真实 observation replanning、direct physical feedback、checkpoint branch recovery、current-state retarget 和简单 state F3 可以组成可靠 simulator 系统。

最终未完成的是 `ACTIONS_AS_COORDINATES_FINAL_METHOD_AND_GOAL.md` 最严格的一条：必须证明 learned action latent 在 execution、refinement、switching 或 recovery 中做了 conventional baseline 不能替代的控制工作。G40 明确给出相反证据，因此不能宣称论文最终目标成功。若未来继续研究，最有价值的方向不是再给同一 task-5 数据加一个 latent head，而是换成具有互相冲突的多动作程序、跨任务多语言变化、不可魔法回滚的执行协议，并让 representation 直接以 held-out causal policy improvement 为目标；同时保留当前 direct/state 系统作为必须击败的强基线。

本程序在这里终止，是因为用户明确把终止点改为完成并审计 EXP_G40，而不是因为最终 latent 目标已经实现。G1–G40 均为产生过新数据、模型、真实 simulator intervention 或机制消融且有 machine-verifiable artifacts 的有效实验；G40 的最终 audit 已通过，磁盘剩余 847 GB。基于证据，最准确的总判断是：**闭环控制系统可行，F2 的 causal feedback 可行，state F3 的显式执行可行；原始 Actions-as-Coordinates latent 贡献在当前数据、policy、benchmark 和 controller interface 下不可行且未被支持。**
