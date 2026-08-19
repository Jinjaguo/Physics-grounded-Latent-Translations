# Next Experiment from EXP_G41: EXP_G42 Local Controllability of the Frozen Action Coordinate

## 当前瓶颈

主瓶颈是 **latent consequence/control semantics**，不是recovery proposal coverage、fallback、F3、failure detection或path hyperparameter。G41中source oracle为7/7，说明所有state都存在place behavior；constrained path显著改善semantic margin和decode consistency，却只有1/7 physical success，且与lift/wrong-language controls没有success优势。当前失败应分类为：planner不知道一个local latent displacement从当前physical state会产生什么TCP、gripper、contact与object effect。

## G42要检验的Actions-as-Coordinates claim

> 在 frozen representation/decoder 附近，是否存在一个低维、state-conditioned、可重复的局部控制子空间，使signed latent-coordinate interventions产生可预测、近似连续且方向一致的真实physical effects；next-language=place是否能选择这个子空间内正确的effect direction？

这不是重新训练policy，也不是recovery proposal生成。实验对象是 frozen latent 的local causal Jacobian/controllability structure。

## 机制变化

从G41的terminal-region path optimization pivot到 **matched signed local interventions + learned action-effect map**：

1. 在train/development lift-boundary states上，围绕真实current latent构造一组有机制差异的方向：decoder Jacobian right-singular directions、empirical local transition tangent directions、Wave21 language-difference direction、random orthogonal controls。
2. 对每个方向执行matched `+epsilon`、`-epsilon` 与zero intervention，只执行一个短decoded prefix；记录真实TCP delta、blue-object delta、gripper/contact change和re-encoded latent。
3. 用train states拟合小型 state-conditioned linear/bilinear effect map `delta_z -> physical effect`，development选择rank与regularization；representation与decoder保持冻结。
4. 做local dimensionality/controllability分析：effect Jacobian有效rank、sign reversal、magnitude monotonicity、跨state direction consistency、decode/re-encode residual与physical smoothness。
5. 在held-out lift states上给next-language=place，仅允许effect map选择一次低维latent control；执行selected-only prefix并与直接LCT direction、G41 constrained path first-step、random direction、action-space effect regression对照。

## 关键因果对照

- `+u` vs `-u`：physical effect是否反向，而不仅是latent distance变化；
- `epsilon`多尺度：effect magnitude是否局部单调/连续；
- language direction vs random orthogonal：place language是否选择更有用的controllable direction；
- frozen latent effect map vs matched direct action-space Jacobian：latent coordinate是否提供额外结构；
- state-conditioned vs state-agnostic：G41失败是否来自同一latent direction在不同contact state产生不同effect。

## Metrics与success rule

主要metrics：signed physical-effect cosine、local linearityR²、effect Jacobian rank、cross-state generalization、decode/re-encode residual、action/TCP/object continuity、lift retention，以及held-out selected prefix的place-progress effect。

只有在held-out上同时满足以下条件才支持claim：

- 至少一个小rank latent subspace产生显著非零、signed可逆、随幅值单调的真实effect；
- state-conditioned effect map跨未见state预测这些effects，优于state-agnostic和random controls；
- place language选择的direction相对wrong-language/random提高真实place-progress且不损害decodability/continuity；
- latent method提供direct action-space baseline没有的可测结构或泛化价值。

若direct action-space map匹配或更好，应接受 frozen latent没有独立control value。若所有latent方向effect高度state-dependent且不可泛化，应报告representation缺少physical state/contact变量，下一步考虑重新定义temporally/physically grounded action coordinate，而不是继续调G41 path loss。

## 不会重复的方向

- 不继续F15 recovery proposal regeneration；
- 不做fallback、safe stop、option graph或failure severity；
- 不重跑Wave28–78 pointwise adapter/threshold/hidden-size搜索；
- 不调整G41 terminal/dynamics/continuity/support权重来救success；
- 不把semantic region entry、EEF distance或offline RedirectGain当作physical controllability；
- 不训练richer-future/world model。

## Required artifacts

G42必须保存preregistration、train/development/held-out split、每个matched `state × direction × sign × magnitude` intervention、decoded action、完整realized feedback、re-encoded latent、effect-map checkpoints、rank/linearity结果、selected-only held-out rollouts、direct-action controls、independent audit、`reports/EXP_G42_report.md`与`reports/next_exp_fromG42.md`。基础设施错误留在G42内解决，不占新ID。
