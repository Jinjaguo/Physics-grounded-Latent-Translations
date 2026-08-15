# Actions as Coordinates
## Project / Paper Story Outline for Codex

## 0. One-Sentence Thesis

**Robot actions can be represented as language-addressable latent coordinates; language can causally redirect the future latent dynamics from the robot's current action state, while trajectory history provides a natural interface for online retargeting, interruption, and return.**

Chinese:

> **机器人动作可以形成可被语言寻址的 latent coordinates；语言能够从当前动作状态出发，因果性地改变未来 latent dynamics 的演化方向，而已经走过的 latent trajectory 则可以进一步作为在线改目标、暂停和返回的路径记忆。**

The project is NOT trying to make a model read one long instruction and autonomously generate an entire multi-stage task plan from scratch.

The project is trying to build a new control interface:

```text
current latent action state
        +
next atomic language goal
        ↓
redirect future latent dynamics
        ↓
decode continuous robot actions
        ↓
optionally retarget / interrupt / return
```

The long-term concept is:

```text
I know where I am in action space.
Language tells me where to go next.
The latent dynamics generate the transition.
I remember how I got here.
Therefore I can change my mind or go back.
```

---

# 1. Core Conceptual Shift

Traditional trajectory-centric view:

```text
observation + long instruction
        ↓
predict a full future trajectory
        ↓
execute
```

Our intended view:

```text
current action coordinate z_t
        +
next language goal g
        ↓
language-conditioned latent dynamics
        ↓
local future trajectory
        ↓
decode and execute
```

Then at any later time:

```text
current latent z_t
        +
new goal g_new
        ↓
redirect again
```

and for return:

```text
stored latent / waypoint history
        ↓
reverse through previously visited recoverable states
```

The system is therefore designed to be:

```text
Addressable
Redirectable
Interruptible
Reversible
```

---

# 2. Main Paper Story

The paper should tell a progression, not a collection of isolated experiments.

## Stage 1 — Actions form usable latent coordinates

We first establish that robot action chunks can be embedded into a continuous latent space that retains two properties:

1. **Language addressability**:
   atomic actions such as

```text
lift_blue_block_slider
place_in_slider
turn_on_lightbulb
...
```

occupy language-addressable latent structure.

2. **Motor executability**:
   those latents can still be decoded back into continuous robot control.

This establishes:

> **Actions can function as coordinates rather than opaque internal variables.**

Important:
This is stronger than ordinary semantic classification because the same latent is still tied to continuous action execution.

---

## Stage 2 — Latent dynamics can evolve continuously

Once action latents are meaningful and executable, study their temporal evolution.

The latent trajectory is treated as continuous across language annotation boundaries.

An annotation change does NOT reset physics.

The language label may change discretely, while physical motion and latent dynamics remain continuous.

Historical result:

```text
latent prediction is possible
long-horizon prediction accumulates drift
iterative refinement can reduce drift in CALVIN
```

The old story was:

> Language anchors action meaning; prediction advances the latent trajectory; refinement stabilizes its continuous evolution.

This remains historical support, but it is no longer the strongest final paper contribution.

---

# 3. Strongest Current Discovery

The most important result from the latest research line is Wave21.

## Key finding

Given the same current/history latent state:

```text
same z_previous
same z_current
same model
same weights
same rollout settings
```

changing only:

```text
next language goal
```

changes the future latent trajectory.

Wave21:

```text
Full RedirectGain = 0.250126
95% CI = [0.136495, 0.370798]

Execution RedirectGain = 0.183855
95% CI = [0.100917, 0.263777]
```

Therefore the strongest defensible statement is:

> **Changing only the next-goal language causally redirects the predicted latent dynamics, including execution-space coordinates.**

This is the central scientific discovery.

Language is no longer only:

```text
a descriptor of action latent
```

It also acts as:

```text
a control variable for the latent vector field
```

Conceptually:

```text
z_{t+1} = F(z_t, g)
```

or, with history:

```text
z_{t+1} = F(z_{t-1}, z_t, g)
```

Changing `g` changes the vector field.

---

# 4. Language as Coordinate vs Language as Attractor

Use these terms carefully.

## Supported idea: language as coordinate / vector-field selector

Language can:

```text
identify action meaning
and
redirect future latent evolution
```

This is supported.

## Not yet fully supported: language as a static attractor

Do NOT write:

```text
each language goal corresponds to one fixed latent attractor point
```

or:

```text
each action is one static executable endpoint subset
```

Waves22–24 show this is too simple.

A better conceptual model is:

> **Language selects a conditioned family of transition dynamics, possibly containing one or more attractor-like structures, rather than specifying one fixed endpoint.**

If a future experiment shows many different initial states converge under the same language-conditioned dynamics to a stable invariant set, then language-conditioned attractor language becomes appropriate.

For now prefer:

```text
language-conditioned vector field
language-selected transition family
language-conditioned dynamical field
```

---

# 5. What Waves22–24 Taught Us

These waves should be framed as mechanism discovery, not a sequence of failures.

## Wave22 — Global decoder consistency is insufficient

Wave21 rollout accumulates encoder-decoder cycle drift.

The frozen cycle map:

```text
C(z) = E(D(z))
```

strongly reduces residual:

```text
2.939692 -> 0.272356
```

but also damages language target identity:

```text
RedirectGain:
0.250126 -> 0.094419
CI crosses zero

endpoint accuracy:
0.516260 -> 0.401423
```

Conclusion:

> **Being globally decoder-consistent is not sufficient to preserve the language-selected target direction.**

A generic "project back to executable manifold" interpretation is too simple.

---

## Wave23 — Goal-specific geometry is explanatory, but static endpoint attraction is not corrective

Goal-core geometry strongly predicts target identity:

```text
goal-core margin / correctness:
Pearson r = 0.7148
Spearman rho = 0.8442

incremental R^2 beyond global cycle residual = 0.425
```

Therefore language goal classes do organize latent geometry.

However, pulling predictions toward a static goal core worsens endpoint identity / decode-reencode identity.

Conclusion:

> **Goal identity is geometrically meaningful, but a goal should not be modeled as one static endpoint set that all trajectories must collapse into.**

---

## Wave24 — Current state predicts transition direction, but averaging destroys trajectory structure

From paired train transitions, source-conditioned local displacement predicts the true future direction:

```text
full cosine = 0.627467
95% CI [0.599828, 0.655551]

execution cosine = 0.647801
95% CI [0.619503, 0.674799]
```

It also beats goal+horizon mean displacement.

Therefore:

> **Current state contains real information about how the future transition should move.**

However deterministic weighted averaging:

```text
underestimates displacement magnitude
worsens endpoint identity
worsens continuity
```

Magnitude recovery is only approximately:

```text
56%–66%
```

Working interpretation:

> Nearby current states may support multiple valid transition modes. Averaging them preserves a rough direction but cancels magnitude and mixes incompatible local motions.

This is a hypothesis to be tested, not yet a proven fact.

---

# 6. Current Working Hypothesis

The current best model is:

> **Language selects a state- and possibly horizon-conditioned distribution / family of latent transition modes. The current latent and recent trajectory determine which particular transition is appropriate.**

Formal object:

```text
p(
    delta_h
    |
    z_previous,
    z_current,
    language_goal,
    horizon
)
```

where:

```text
delta_h = z_future_h - z_current
```

This is more appropriate than:

```text
language -> one endpoint point
```

or:

```text
language -> one static goal core
```

or:

```text
language -> one deterministic mean displacement
```

---

# 7. Wave25 Research Role

Wave25 is the distributional-mechanism decision point.

It should test:

```text
Are the paired transition displacements multi-modal?

Does mean cancellation explain Wave24 magnitude shrinkage?

If the correct mode is known by an oracle,
does endpoint identity / decoded action / continuity improve?

Can a causal non-oracle selector infer the correct mode
from z_previous + z_current + language + horizon?
```

Compact implementation candidates:

```text
nearest-source mode
KNN mode vote
regularized logistic mode selector
small MLP selector
compact mixture-density head
```

Do not jump directly to a large diffusion/flow model with only ~257 train transitions.

If compact mixture structure works:
move to mode-aware latent dynamics.

If oracle works but causal selector fails:
the latent state probably lacks phase/contact information.

If oracle itself fails:
a discrete mixture is too crude; test a continuous latent distribution such as diffusion / flow matching.

---

# 8. Implementation Ladder

Do not treat each failed implementation as a reason to weaken the central claim.

The correct mindset is:

```text
Scientific fact:
language redirects latent dynamics
        ↓
Implementation question:
what transition model best realizes that signal?
```

Implementation ladder:

```text
deterministic prediction
        ↓
local deterministic displacement
        ↓
compact multimodal displacement
        ↓
mixture-density / mixture-of-experts
        ↓
conditional latent diffusion / flow
        ↓
phase/contact-conditioned latent state
```

Different implementations may fail while the causal language-direction finding remains valid.

---

# 9. Online Retargeting: Major Future Capability

The paper/system story should add one important layer beyond offline transition prediction.

## Online retargeting

At any time:

```text
current latent = z_t
old goal = g_old
```

the user/operator can provide:

```text
new goal = g_new
```

Then:

```text
z_t + g_new
    ↓
new language-conditioned latent dynamics
    ↓
new future trajectory
```

The system does not need to commit to a complete long trajectory at the beginning.

Example:

```text
robot is holding a lifted object
        ↓
user: place_in_slider
        ↓
latent dynamics begin place transition

halfway through:
user gives another atomic goal
        ↓
trajectory is redirected from the current latent
```

The key property is:

> **The robot can revise its future without restarting the task representation from the beginning.**

Use wording such as:

```text
online retargeting
incremental goal specification
interruptible latent dynamics
receding language-conditioned control
```

Avoid claiming that ordinary VLA/WAM systems are incapable of replanning.
The distinction is the interface and representation, not an absolute impossibility claim.

---

# 10. Trajectory Memory and Return

The second major future capability is `stop / return`.

During execution, store:

```text
z_0
z_1
z_2
...
z_t
```

and preferably:

```text
decoded action chunks
robot/simulator waypoint metadata
timestamps
recoverability markers
```

Then if the user issues:

```text
STOP
RETURN
```

the system can use the stored execution history.

Minimal implementation:

```text
history stack:
[z_0, z_1, ..., z_t]

return:
z_t -> z_{t-1} -> ... -> z_0
```

However:

**do not claim strict physical time reversal.**

Robot/environment dynamics may be irreversible due to:

```text
contacts
release
collisions
switching
drawer mechanisms
object drops
gripper state changes
```

Safer wording:

> **Return to a previously visited recoverable state along stored latent/physical waypoints.**

The easiest initial demonstration:

```text
lift_blue_block_slider
        ↓
place_in_slider begins
        ↓
execute 1–2 latent steps
        ↓
RETURN
        ↓
return to stored post-lift state
        ↓
place_in_slider again
        ↓
continue
```

This provides:

```text
interruptibility
retargetability
history-aware reversibility
```

---

# 11. The Full Intended System

The final system concept is:

```text
                 ┌──────────────────────┐
                 │ next language goal g │
                 └──────────┬───────────┘
                            │
                            v
previous latent -----> current latent z_t
                            │
                            v
              language-conditioned
                  latent dynamics
                            │
                            v
                  future latent chunk
                            │
                            v
                     action decoder
                            │
                            v
                  continuous control
                            │
                            v
                      environment
                            │
                            v
                      new latent state
                            │
               ┌────────────┴────────────┐
               │                         │
           continue                  new language
                                         │
                                         v
                                     retarget
```

In parallel:

```text
trajectory history buffer
[z_0, z_1, ..., z_t]

        ↓ STOP / RETURN

recover previous waypoint(s)
```

---

# 12. Desired Final Contribution Structure

The ideal mature paper has four contribution levels.

## Contribution 1 — Language-addressable action coordinates

Action chunks form a continuous latent space that is:

```text
semantically addressable
and
motor-decodable
```

## Contribution 2 — Language-conditioned latent dynamics

Changing only the next language goal changes future latent dynamics from the same current state.

This is the strongest current causal result.

## Contribution 3 — Distributional state-dependent transition structure

Future work / Wave25+:

```text
language chooses a transition family
current latent/history chooses a local mode
```

This is currently being tested.

## Contribution 4 — Interactive trajectory editing

Future closed-loop system:

```text
retarget anytime
interrupt anytime
return through history
```

This is an intended systems contribution, not yet experimentally established.

---

# 13. Safe Current Claims

The following are currently defensible.

### Claim A

> **Robot action chunks can be organized into language-addressable latent coordinates while preserving continuous action decodability.**

### Claim B

> **Changing only the next-goal language causally redirects predicted latent dynamics from the same current action state.**

### Claim C

> **The language-induced change extends into execution-space coordinates.**

### Claim D

> **Current-state-conditioned train transitions contain predictive information about future displacement direction beyond goal/horizon averages.**

### Claim E

> **Global decoder consistency and static goal-core attraction are insufficient to recover executable target transitions.**

### Claim F

> **Goal-specific train geometry predicts target identity beyond global encoder-decoder cycle consistency, but static endpoint attraction is not an adequate transition mechanism.**

---

# 14. Claims Not Yet Authorized

Do NOT currently claim:

```text
language is a proven stable attractor

each language goal is one fixed executable endpoint subset

language is already an executable target coordinate

goal-core alignment solves transition execution

state-conditioned multimodality is proven

closed-loop goal switching works

return-to-history works physically

the system autonomously plans arbitrary long-horizon tasks

the system is strictly stronger than all VLA/WAM systems
```

These remain hypotheses / future capabilities.

---

# 15. How to Position Against VLA / WAM

Do not say:

> "VLA/WAM predicts a full trajectory once and cannot change it."

That is too broad and easy to rebut.

Instead say:

> **Our goal is a different control abstraction. Rather than treating each new instruction as a request to regenerate an entire future behavior, we expose the current action latent as an editable dynamical state. Language modifies the local future dynamics from that state, while stored trajectory history supports interruption and return.**

The contrast is:

```text
trajectory generation interface
vs
editable latent dynamical interface
```

not:

```text
they cannot replan
vs
we can
```

---

# 16. Core Paper Vocabulary

Preferred terms:

```text
Actions as Coordinates

language-addressable action coordinates

language-conditioned latent dynamics

causal language redirection

execution-space redirection

state-conditioned transition family

conditional transition distribution

online latent retargeting

trajectory memory

interruptible execution

return-to-history

recoverable waypoint
```

Use carefully:

```text
attractor
manifold
executable manifold
```

Only when experimentally justified.

Prefer:

```text
empirical executable region
decoder-supported coordinates
transition family
latent vector field
```

---

# 17. Central Figures for the Final Paper

## Figure 1 — Main concept

Show one current latent with several language commands:

```text
same current z_t

"place_in_slider" -> trajectory A
"turn_on_lightbulb" -> trajectory B
"push_pink_block_right" -> trajectory C
```

Same start, different language, different future vector fields.

## Figure 2 — Actions as coordinates

Show language/action latent organization and decoder executability.

## Figure 3 — Causal language swap

Quantitative RedirectGain / execution RedirectGain.

## Figure 4 — Why endpoint attraction fails

Show:

```text
global cycle projection
static goal-core attraction
deterministic displacement averaging
```

and the failure patterns.

## Figure 5 — Transition distribution

Future Wave25+:

```text
same language
different current states
different transition modes
```

## Figure 6 — Online retargeting

```text
lift
 -> place
 -> mid-course new goal
 -> redirect
```

## Figure 7 — Return-to-history

```text
forward trajectory
interrupt
reverse/recover through stored waypoints
```

---

# 18. Candidate Paper Titles

Current strong options:

### Option 1
**Actions as Coordinates: Language-Conditioned Latent Dynamics for Redirectable Robot Control**

### Option 2
**Actions as Coordinates: Language-Addressable and Redirectable Latent Dynamics for Robot Manipulation**

### Option 3
**Language as a Control Coordinate for Latent Robot Dynamics**

### Option 4
**Editable Latent Robot Dynamics: Language-Conditioned Action Coordinates with Retargeting and Return**

Avoid using "variational" in the title.
The variational-mechanics hypothesis was rejected historically.

---

# 19. Candidate Abstract-Level Story

A future abstract can follow this structure:

1. Modern robot policies generate actions from observations and language, but the internal action representations are rarely exposed as an editable control interface.

2. We ask whether continuous action latents can instead function as coordinates: semantically addressable by language, decodable into robot control, and dynamically steerable from the current action state.

3. We construct a language-grounded action latent representation and show that atomic manipulation actions occupy addressable yet executable latent structure.

4. Crucially, holding the current action state fixed while changing only the next-goal language causally redirects the predicted future latent trajectory, including its execution subspace.

5. Mechanistic experiments show that this redirection cannot be reduced to projection onto a global decoder-supported set or attraction toward a static goal endpoint; current-state-conditioned transition geometry carries additional predictive information.

6. We therefore model language-conditioned manipulation as an editable latent dynamical system in which language selects the next transition family while the current latent specifies the local trajectory.

7. This abstraction naturally supports future interactive control primitives such as online retargeting, interruption, and return through stored trajectory history.

Do not include future capabilities as completed experimental results until they are actually demonstrated.

---

# 20. Project Philosophy

The project should not be optimized around making one specific architecture win.

The research question is:

> **What structure turns a learned robot latent into a programmable control interface?**

The sequence of experiments is valuable because each failed simple model identifies what structure is missing:

```text
semantic latent only
    ↓
need executability

deterministic dynamics
    ↓
need stability

global executable projection
    ↓
target geometry mismatch

static target core
    ↓
missing transition correspondence

deterministic local displacement
    ↓
possible mode averaging

distributional transition model
    ↓
current direction
```

The paper should present this as systematic mechanism discovery, not as post-hoc rescue.

---

# 21. Long-Term Vision

The final vision is broader than CALVIN.

A foundation model latent should become an interface that can be:

```text
read
addressed
steered
retargeted
interrupted
replayed
reversed/recovered
constrained by physics
```

For embodied intelligence, this means the robot should not need to regenerate its entire future whenever the user changes intent.

Instead:

> **The robot maintains an internal action coordinate, language changes where the dynamics should go next, and trajectory memory preserves how it arrived there.**

A concise slogan:

> **Know where you are. Tell the robot where to go next. Remember how you got there.**

Or:

> **Actions are coordinates; language edits the future.**

---

# 22. Instructions for Codex

When working on future waves:

1. Read this file first.
2. Preserve all historical claim decisions.
3. Never rewrite a rejected gate as supported.
4. Do not treat future capabilities as completed results.
5. Keep the Wave21 causal language-redirection result central.
6. Prefer experiments that reveal the correct transition structure over repeated threshold tuning.
7. Avoid reopening DEL / variational-mechanics rescue in this project line.
8. Avoid repeatedly adding static endpoint-attraction losses.
9. Maintain source-session statistical independence.
10. Preserve held-out discipline.
11. Use implementation exploration on train/development when the scientific question is model form.
12. Once the transition model becomes reliable, prioritize:
    - online retargeting;
    - interruption;
    - return-to-history;
    - closed-loop matched-state evaluation.
13. Do not claim strict physical reversibility without testing it.
14. Distinguish:
    - latent waypoint reversal;
    - recoverable-state return;
    - physical time reversal.
15. The final system should remain incremental:
    - current state;
    - next language;
    - local transition;
    - optional new language;
    - optional return.

This file is the current project-level story and should be updated only when a new experiment materially changes the scientific interpretation.
