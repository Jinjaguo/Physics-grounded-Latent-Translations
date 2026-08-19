# Next Experiment from EXP_G42: EXP_G43 Coordinate Geometry or Decoder Parameterization?

## 当前瓶颈

瓶颈是 **representation-specific control value**。G42证明latent perturbation能通过decoder产生局部physical effects，但random directions同样单调，direct decoded-action effect prediction更好。尚未回答：这些性质来自learned language/action geometry，还是任何可逆重参数化后的decoder输入都会有。

这不是proposal coverage、recovery、fallback、handoff或F3问题。

## G43 scientific claim

> 如果action latent是真正的coordinate system，而不只是decoder的任意参数化，那么语言/transition geometry与locally controllable physical subspace之间应存在可跨state保持的对齐；这种对齐应在破坏semantic geometry但保持decoder function的invertible reparameterization controls下消失，并对goal-conditioned intervention产生可测影响。

## 机制实验

构造function-preserving coordinate controls。对frozen latent应用固定invertible transforms `z'=Rz`，同时把decoder第一层改写为`D'(z')=D(R^{-1}z')`，确保decoded action function逐点相同。比较：

1. 原始learned coordinate；
2. random orthogonal rotation；
3. whitening/axis-rescaling coordinate；
4. semantic/execution block-preserving rotation；
5. matched direct action coordinate。

这些controls保持可执行function不变，只改变coordinate geometry。随后在new session states上比较：

- language-region direction与physical controllable subspace的principal-angle alignment；
- transition-PCA basis跨state的parallel consistency；
- low-rank effect-map sample efficiency与cross-state R²；
- 从next-language生成一次selected-only local intervention时的真实place-progress effect；
- random/wrong-language matched controls。

如果rotation后性能不变，所谓优势来自decoder function而不是learned axes，应删除“learned coordinate geometry”claim。如果只有原始semantic geometry在低数据下更好，才支持language-addressable coordinates具有独立control value。

## 数据与held-out discipline

不能复用G41/G42 held-out states作为新test。应从Wave21 inventory中选择尚未进入G41/G42的独立source sessions，或扩展到`lift_red_block_table -> place_in_slider`并按source session重新划分。所有reparameterizations、rank、effect target和selection rule必须在train/development冻结后一次性打开held-out。

## Success rule

支持claim需要同时满足：

- 原始coordinate的semantic/control alignment显著高于full random rotation，但block-preserving control保留预期结构；
- 原始coordinate在matched low-rank/data budget下提高held-out physical-effect prediction或selected place-progress；
- direct action baseline不能完全解释该增益；
- function-preserving transform的decoded action identity经数值与执行验证。

若原始与rotated coordinate相同，或direct action仍更强，应接受latent只是decoder parameterization，不继续增加adapter、loss、threshold或state heads。

## 不重复方向

- 不恢复F15 recovery proposal、fallback或option graph；
- 不调G41 path weights/horizon/terminal thresholds；
- 不给G42 bilinear state model加更多features来救held-out；
- 不重跑Wave28–78 low-rank pointwise tournament；
- 不把动作变化、latent rank或semantic classification单独当作control-coordinate成功。

## Required artifacts

保存每个transform的精确定义、function-preserving decoder rewrite、identity audit、session split、matched interventions、principal angles、effect-map checkpoints、selected-only physical rollouts、direct-action controls、independent audit、`reports/EXP_G43_report.md`与`reports/next_exp_fromG43.md`。
