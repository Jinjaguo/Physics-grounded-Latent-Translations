# EXP_G42 Report: Local Physical Controllability of a Frozen Action Coordinate

## 结论

**SUPPORTED：frozen latent neighborhood通过decoder具有低维、近似连续的真实physical effect。NOT SUPPORTED：这种local controllability来自learned semantic action-coordinate geometry，或比decoded action本身提供更强的跨state控制接口。**

G42在10个此前未进入G41 lift→place pair的独立CALVIN source sessions上执行了790条fresh-env matched rollouts：5 train states、2 development states、3 prospective held-out states；每个state包含zero baseline和decoder-Jacobian、empirical transition-PCA、place-minus-lift language-region、random-orthogonal四类signed multi-magnitude latent interventions。总计执行12,640 simulator steps，representation与decoder从未更新。

在held-out上，四类directions都产生2–4维physical effect、0.833–0.958的magnitude monotonicity与有限signed antisymmetry。说明latent displacement并非任意噪声：通过frozen decoder，它能连续改变TCP和blue-object trajectory。然而random方向的monotonicity最高0.958，structured directions没有独占这种性质。更关键地，cross-state effect prediction中latent-only R²为0.574，明显低于直接用decoded action delta的0.754。learned latent没有证明独立control value。

## 科学问题

G41发现language-conditioned latent path能显著改变真实trajectory并改善semantic region/cycle consistency，但不能可靠产生place。G42不再调整G41 path loss，而直接问：

> frozen action latent附近是否存在一个小维度、跨state一致的local control subspace，使signed latent interventions产生可预测、随幅值连续的真实physical effects；该结构是否优于random latent directions和matched decoded-action coordinates？

这不是recovery proposal、fallback、failure selector或task-success优化。每条intervention只改变current frozen action latent并执行其decoded 16-step prefix。

## 数据与split

从Wave21 inventory中选择`previous_label=lift_blue_block_slider`、`next_label!=place_in_slider`的states，避免重用G41的21个lift→place pairs。每个session取第一条eligible boundary：

- train sessions：4、6、11、12、13；
- development sessions：3、15；
- held-out sessions：16、26、28。

这10个sessions都未出现在G41 pair cohort。physical `robot_obs/scene_obs`通过官方CALVIN ZIP member range extraction取得；held-out physical frames在direction families、rank=4、magnitudes、ridge与effect models全部冻结后才打开。

## Interventions

current latent由boundary前最后16个真实recorded actions编码。train-only one-step latent displacement的median norm定义统一scale。每个非zero direction执行：

- signs：-1、+1；
- relative magnitudes：0.25、0.5、1.0；
- decoded prefix：16个CALVIN controls；
- matched zero：从同一fresh physical state执行`decode(z_current)`。

方向families：

1. `decoder_jacobian`：current latent处frozen decoder continuous-action Jacobian的前4个right singular directions；
2. `transition_pca`：Wave21 train one-step displacements的前4个PCA directions；
3. `language_region`：train-only place-region centroid minus lift-region centroid，一维；
4. `random_orthogonal`：每个session预注册seed产生的4个正交方向。

physical effect定义为intervention相对zero的TCP endpoint delta与blue-object endpoint delta，共6维。所有effect来自真实CALVIN执行，不是decoder prediction。

## Direction-level results

| Family | Development monotonic | Held-out monotonic | Development antisymmetry error ↓ | Held-out antisymmetry error ↓ | Held-out effect rank | Held-out mean effect norm |
|---|---:|---:|---:|---:|---:|---:|
| decoder Jacobian | 0.946 | 0.875 | 0.278 | **0.305** | 3 | **0.1023** |
| transition PCA | 0.982 | 0.917 | **0.250** | 0.331 | **4** | 0.0585 |
| language region | **1.000** | 0.833 | 0.265 | 0.334 | 2 | 0.0456 |
| random orthogonal | **1.000** | **0.958** | 0.357 | 0.382 | 3 | 0.0451 |

antisymmetry error为0才表示`+u/-u`完全相反。所有families约0.30–0.38，说明局部effect只是近似signed，并受contact/nonlinearity影响。decoder-Jacobian方向产生最大effect，transition-PCA覆盖最大held-out rank；language direction没有相对random的明确优势。

## Cross-state effect prediction

三种ridge maps使用同一train interventions与相同6-D physical target：

| Predictor | Train R² | Development R² | Held-out R² |
|---|---:|---:|---:|
| latent delta | 0.831 | 0.806 | 0.574 |
| state-conditioned latent bilinear | 0.883 | 0.823 | **-32818.183** |
| decoded action delta | **0.960** | **0.898** | **0.754** |

latent delta确实含有可泛化effect信息，不能说latent完全无控制结构。但direct decoded action在三个split都更好，因此现有证据支持“decoder action mapping可控”，不支持“learned latent geometry提供额外effect interface”。

state-conditioned model只比latent在development增加0.016 R²，却在held-out灾难失败。机制诊断发现5个train state PCA codes最大绝对值5.96；一个held-out state因raw state维度超出train support，code达到5024。该模型预测effect norm均值7.45、最大74.1，而真实均值0.0669、最大0.304。这否定当前5-state bilinear conditioning，不是增加feature或regularizer继续救它的理由。

## Claim decisions

- **SUPPORTED：** 通过frozen decoder，local latent directions能产生非零、随幅值大体单调、2–4维的真实physical effects。
- **SUPPORTED：** decoder-Jacobian与transition-PCA提供不同effect magnitude/rank结构。
- **NOT SUPPORTED：** language-region direction比random方向更可控或更稳定。
- **NOT SUPPORTED：** learned latent比decoded action提供更好的cross-state physical-effect prediction。
- **NOT SUPPORTED：** 当前naive state-conditioned bilinear map能泛化到unseen physical states。
- **NOT TESTED：** 这些local directions能否组成多step goal-correct lift→place transition；G41已显示当前path formulation不能，G42只做local system identification。

## 与G41相比的新信息

G41只能说language-conditioned path改变了trajectory但方向不对。G42定位了更底层的原因：latent neighborhood不是完全不可控，局部effect相当规则；但这种规则大部分可由frozen decoder的action mapping解释，semantic/language geometry没有对physical controllability提供可测优势。下一步科学问题不应是再调path optimizer，而应区分“learned coordinate structure”和“任意decoder parameterization”。

## Bugs、无效runs与结论影响

- 2-state smoke用rank=1/magnitude=0.5验证Jacobian、simulator与effect-map接口，不进入正式aggregate。
- 正式development为546 nonzero + 7 zero；held-out为234 nonzero + 3 zero，artifacts数量均完整。
- state-conditioned held-out极端负值经独立重算确认，不是JSON、normalization或replay错误。
- independent audit重新编码234 latents并重放全部237 held-out rollouts，共3,792 steps；maximum re-encode、effect与physical endpoint replay error均为0。
- 完成后磁盘剩余约845 GB。

## Machine-verifiable artifacts

- Preregistration: `experiments/EXP_G42_preregistration.json`
- Runner: `scripts/experiments/run_exp_g42_local_controllability.py`
- Auditor: `scripts/experiments/audit_exp_g42.py`
- Development data/results: `experiments/EXP_G42_development/boundary_frames/`, `run_default/`
- Frozen effect maps: `experiments/EXP_G42_development/run_default/effect_models.npz`
- Held-out states: `experiments/EXP_G42/boundary_frames/source_manifest.json`
- Held-out interventions/rollouts: `experiments/EXP_G42/heldout/interventions.jsonl`, `rollouts/`
- Held-out metrics/audit: `experiments/EXP_G42/heldout/metrics.json`, `audit.json`

## 最终判断

G42支持一个窄结论：frozen action latent可作为decoder输入的local perturbation coordinate，并具有可测的低维physical response。它不支持论文更重要的结论：learned language-addressable geometry本身就是更好的programmable control coordinate。只要random directions同样单调且direct action effect map持续更强，就必须把“coordinate value”置于prove-or-drop，而不是因为latent能改变动作就宣布成功。
