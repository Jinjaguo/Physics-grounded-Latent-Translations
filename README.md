# PGLT representation release

This repository contains the paper-release implementation of the PGLT
language-grounded CALVIN action representation. Representation search is
complete: the frozen prospective replication passed on all 14 independent
official episodes, all six seeds, both retrieval directions, and all 84 motor
cells.

The historical all-cell confirmation gate remains recorded as `FAIL`; the
separate prospective Representation Readiness Gate is `PASS`. The latter
freezes the representation for the paper and authorizes the next latent
dynamics stage without erasing the historical failed cell.

## Project north star: Actions as Coordinates

The project-level paper story, supported claims, rejected interpretations, and
long-term system target are maintained in
[`actions_as_coordinates_project_paper_outline.md`](actions_as_coordinates_project_paper_outline.md).

**Required pre-run protocol:** before planning or executing every new
experimental wave, collaborators and coding agents must read that outline in
full and align the experiment with it. In particular, keep the Wave21 causal
language-redirection result central, preserve all historical claim decisions,
do not present distributional transition structure or interactive capabilities
as established before their gates pass, and prefer experiments that identify
the missing transition structure over repeated threshold or attraction-loss
tuning.

The outline is the research north star; frozen manifests, preregistrations, and
recorded experimental results remain the source of truth for exact protocol and
claim status. Update the outline only when a new experiment materially changes
the scientific interpretation.

After the Wave28–Wave78 pointwise steering program reached its registered upper
bound, the active research direction is hierarchical latent path planning. The
post-Wave78 specification is [`prompts/ACTIONS_AS_COORDINATES_POST_WAVE78_RESEARCH_DIRECTION.md`](prompts/ACTIONS_AS_COORDINATES_POST_WAVE78_RESEARCH_DIRECTION.md), and the first registered protocol is [`prompts/EXP_R1_AUTONOMOUS_LATENT_HIERARCHICAL_CONTROL_CODEX_PROMPT.md`](prompts/EXP_R1_AUTONOMOUS_LATENT_HIERARCHICAL_CONTROL_CODEX_PROMPT.md).
Future experiments use the `EXP_R1`–`EXP_R80` namespace; Wave79 is not a valid
experiment name. `EXP_R81` is forbidden. The initial protocol freezes the accepted representation,
decoder, and F1, retains the old F2 as a baseline, uses oracle F3 boundaries,
and tests multi-step latent path planning before learned task completion.

The post-Wave78 open-loop program reached `SUPPORTED` at EXP_R8 with
`repair_late_0.75`.  The subsequent bounded closed-loop stage EXP_R9–EXP_R58
then tested latent replay, action-conditioned/history plants, robust shocks,
target-set capture, completion readiness, and waypoint/return interface gates.
It did not establish the full physical closed-loop system because exact
simulator/controller snapshots are absent.  EXP_R58 is complete and EXP_R59
is forbidden.

## Released method

- Input: an action-only `H=16` CALVIN `rel_actions` window.
- Latent: 32 dimensions, split into 16 semantic and 16 execution coordinates.
- Decoder: reconstructs all seven action channels from the full latent.
- Text: frozen OpenCLIP ViT-L/14 DataComp-XL features with a trainable linear
  projection to the semantic subspace.
- Gradient rule: language gradients update the semantic head and text
  projection but are detached from the shared action trunk.
- Objective: motor reconstruction plus symmetric contrastive alignment;
  shuffled-language and reconstruction-only controls are capacity matched.
- Optimization: AdamW, 40 epochs, EMA decay 0.999, six registered seeds.

## Repository layout

- `src/pglt/representation/`: final model, dataset, objectives, EMA, retrieval,
  and readiness-gate implementation.
- `scripts/representation/`: functional data verification, validation,
  training, evaluation, aggregation, audit, and orchestration entry points.
- `configs/representation.yaml`: the single released scientific config.
- `data/representation/`: 31 compact official training episodes, metadata,
  and frozen text features.
- `checkpoints/representation/`: 18 frozen epoch-40 EMA checkpoints and hashes.
- `results/representation/`: development evidence, historical gate evidence,
  independent replication metrics, and release integrity.
- `paper/representation_iclr2026.tex`: ICLR 2026-style representation draft.
- `actions_as_coordinates_project_paper_outline.md`: project/paper north star;
  required reading before every new experimental wave.
- `archive/representation_development/`: local, Git-ignored development
  prompts, logs, manifests, and iteration reports. It is not imported by
  released code.
- `src/pglt/dynamics/`, `scripts/dynamics/`: isolated next-stage interfaces and
  numerical software-validation utilities; they do not modify representation.
- `third_party/`: optional, Git-ignored upstream CALVIN and LaWM sources used
  only for provenance or the separate dynamics diagnostics.

## Environment

Create a Python 3.10+ environment and install the package:

```bash
pip install -e ".[test]"
```

GPU training/evaluation additionally requires a CUDA-enabled PyTorch build.
The original runs used Python 3.10, PyTorch 2.10.0+cu128, and an RTX 4090.

## Reproduction

Checkpoint-only reproduction (no training):

```bash
python scripts/representation/reproduce.py \
  --config configs/representation.yaml \
  --stage verify evaluate summarize test audit \
  --device cuda:0
```

Full reproduction, including 8-fold development validation and six-seed final
training:

```bash
python scripts/representation/reproduce.py \
  --config configs/representation.yaml \
  --stage verify validate train evaluate summarize test audit \
  --device cuda:0
```

Individual stages can also be run directly. Every new script documents its
parameters, usage, and outputs in the opening docstring.

## Final representation result

- Independent episodes with positive six-seed mean in both directions: 14/14.
- Seeds with positive episode mean in both directions: 6/6.
- Whole-episode bootstrap lower 95% bounds: 0.7992 T→A, 0.6606 A→T.
- Motor cells passing the 20% MSE / 0.05 gripper limits: 84/84.
- Negative independent semantic cells: 0.
- Worst relative continuous-MSE increase: 4.16%.
- Worst gripper-accuracy drop: 0.00329.

See `results/representation/independent_replication/summary.md` and
`paper/representation_iclr2026.tex` for the full account.

## Paper draft

The TeX source uses the official ICLR 2026 style files included in `paper/`.
Compile on a machine with a LaTeX distribution:

```bash
cd paper
pdflatex representation_iclr2026.tex
pdflatex representation_iclr2026.tex
```

The current development environment does not include `pdflatex`; the source
has been statically checked for balanced environments and braces.
