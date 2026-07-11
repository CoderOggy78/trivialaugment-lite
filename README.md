# trivialaugment-lite

A clean, dependency-light PyTorch reimplementation of **TrivialAugment (TA)** — the
parameter-free data augmentation baseline from Müller & Hutter, *"TrivialAugment: Tuning-free
Yet State-of-the-Art Data Augmentation"* (arXiv:2103.10158).

TA samples **one** augmentation and **one** random strength per image, applies it once, and
does no policy search whatsoever — yet it matches or beats AutoAugment, RandAugment, Fast
AutoAugment and Adversarial AutoAugment across CIFAR-10, CIFAR-100, SVHN and ImageNet.

```
┌─────────┐    sample op    ┌──────────────┐   sample m ~ U{0..30}   ┌───────────┐
│  image  │ ──────────────► │  augmentation │ ──────────────────────► │ a(x, m)   │
└─────────┘   a ~ U(A)      └──────────────┘                          └───────────┘
```

---

## Table of contents

- [Why TrivialAugment](#why-trivialaugment)
- [How it actually works](#how-it-actually-works)
- [Install](#install)
- [Quick start](#quick-start)
- [Package internals](#package-internals)
- [The augmentation space in detail](#the-augmentation-space-in-detail)
- [Training on CIFAR-10 / CIFAR-100](#training-on-cifar-10--cifar-100)
- [Model zoo](#model-zoo)
- [Extending the augmentation space](#extending-the-augmentation-space)
- [Reference results](#reference-results)
- [Why it works: the averaging-of-distributions argument](#why-it-works-the-averaging-of-distributions-argument)
- [Ablations worth knowing](#ablations-worth-knowing)
- [Compute cost comparison](#compute-cost-comparison)
- [Design decisions in this implementation](#design-decisions-in-this-implementation)
- [Known limitations](#known-limitations)
- [FAQ](#faq)
- [Project layout](#project-layout)
- [Citation](#citation)
- [License](#license)

---

## Why TrivialAugment

Every automatic augmentation method before TA spends compute *searching* for a policy:

| Method | What it searches for | Search cost vs. one training run |
|---|---|---|
| AutoAugment (AA) | RNN-parameterized sub-policies, trained with RL | ~15,000× (>half a GPU-year) |
| AWS | Distribution over augmentation pairs | high, shared weights |
| PBA | Population of evolving policies | tens of workers × training |
| OHL | RL over augmentation-pair distribution | 8 parallel workers |
| Adversarial AA | RL policy that maximizes training loss | 8× (batch augmentation) |
| RandAugment (RA) | Two scalars, `n` and `m`, via grid search | up to 80× |
| Fast AutoAugment | Policies found by inference-time search on splits | ~1× |
| **TrivialAugment** | **nothing** | **0×** |

TA removes the search entirely by making both choices — *which* augmentation and *how
strong* — a fresh uniform random draw for **every image, every epoch**. The paper's central,
somewhat uncomfortable finding is that this parameter-free baseline is competitive with, and
frequently better than, every method above it in the table. This repository exists to make
that baseline trivially easy to drop into a real training pipeline.

## How it actually works

```
procedure TA(x):                      # Algorithm 1 in the paper
    a ← sample uniformly from A        # e.g. rotate, solarize, shear-x, ...
    m ← sample uniformly from {0..30}  # strength, resampled every call
    return a(x, m)
```

Two properties matter more than they look:

1. **Strength is resampled per image, not tuned once for the dataset.** This is the actual
   difference from RandAugment, which uses a single, globally-tuned strength `m` for the whole
   run. TA is *not* a special case of RA — sampling `m` per image changes the effective
   augmented-data distribution qualitatively, not just its variance.
2. **Only one operation is applied.** No stacking, no chaining, no leave-out probabilities.
   This keeps augmented images closer to the original data manifold on average, which the
   paper argues is important — see [Why it works](#why-it-works-the-averaging-of-distributions-argument)
   below.

## Install

```bash
pip install -r requirements.txt
```

Only `torch`, `torchvision`, and `pillow` are required. `demo.py` needs nothing but `pillow`
— you can inspect the augmentation before installing torch at all.

## Quick start

Generate a grid of augmented samples without touching torch:

```bash
python demo.py
```

Use TA as a torchvision-compatible transform:

```python
from torchvision import transforms as T
from trivial_augment import TrivialAugment

train_transform = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode="reflect"),
    T.RandomHorizontalFlip(),
    TrivialAugment(),       # <- drop-in, zero hyperparameters
    T.ToTensor(),
    T.Normalize(mean, std),
])
```

Train a WideResNet on CIFAR-10:

```bash
python src/train.py --dataset cifar10 --model wrn_40_2 --epochs 200
```

## Package internals

`src/trivial_augment.py` is intentionally small — the whole method is one class:

```python
class TrivialAugment:
    def __init__(self, augmentation_space=None, num_strengths=31):
        self.augmentation_space = augmentation_space or RA_SPACE
        self.num_strengths = num_strengths

    def __call__(self, img):
        op = random.choice(self.augmentation_space)
        strength = random.randint(0, self.num_strengths - 1)
        return op(img, strength)
```

Every operation has the signature `op(img: PIL.Image, level: int) -> PIL.Image`, where `level`
is an integer in `[0, 30]`. Each op internally maps that integer level onto its own natural
range via two small helpers:

```python
_float_param(level, maxval)  # -> level / 30 * maxval, e.g. rotate degrees, color magnitude
_int_param(level, maxval)    # -> integer-valued version, e.g. posterize bits, translate pixels
```

This mirrors how AutoAugment/RandAugment discretize continuous parameters into 31 uniformly
spaced levels, which keeps the augmentation space directly comparable to prior work and easy
to swap into existing RA/AA codebases.

## The augmentation space in detail

The default space (`RA_SPACE`) is the 14-operation RandAugment space, matching the "RA" space
used as TA's primary setting in the paper:

| Operation | Effect | Parameter range at max strength |
|---|---|---|
| `identity` | no-op | — |
| `auto_contrast` | stretches the pixel histogram | — |
| `equalize` | histogram equalization | — |
| `rotate` | rotates by ±θ | ±30° |
| `solarize` | inverts pixels above a threshold | threshold 256→0 |
| `color` | saturation enhance/reduce | 0.1–1.9 |
| `posterize` | reduces bits per channel | 8→4 bits |
| `contrast` | contrast enhance/reduce | 0.1–1.9 |
| `brightness` | brightness enhance/reduce | 0.1–1.9 |
| `sharpness` | sharpness enhance/reduce | 0.1–1.9 |
| `shear_x` / `shear_y` | affine shear | 0.0–0.3 |
| `translate_x` / `translate_y` | affine translate | 0–image_size/3 px |

Several ops are **randomized in direction** as well as strength (rotate, color, contrast,
brightness, sharpness, shear, translate all flip sign with probability 0.5), which matches the
reference implementation and roughly doubles the effective diversity of each operation without
adding parameters.

Notably absent from the default space: `invert`, `cutout`, horizontal/vertical `flip`, `blur`,
`smooth`, `sample_pairing`. The paper's ablation (Table 6) found the *Full* space, which
includes all of these, actually performs **worse** than the smaller RA/Wide/AA spaces — bigger
isn't better, and a hand-picked space still matters more than the search process.

## Training on CIFAR-10 / CIFAR-100

```bash
python src/train.py \
    --dataset cifar100 \
    --model wrn_28_10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --weight-decay 5e-4
```

The training loop follows the paper's CIFAR recipe: SGD with Nesterov momentum, cosine
learning-rate decay, and, on top of TrivialAugment, the same fixed pre/post-processing used
in the paper — random crop with reflect padding, random horizontal flip, and a final 16×16
cutout. Checkpoints are written continuously to `checkpoints/last.pt` and `checkpoints/best.pt`
(best by held-out test accuracy) so a run can be resumed or evaluated mid-training.

Evaluate any checkpoint independently:

```bash
python src/evaluate.py --dataset cifar100 --model wrn_28_10 --checkpoint checkpoints/best.pt
```

## Model zoo

`src/model.py` implements a standard pre-activation WideResNet (Zagoruyko & Komodakis, 2016),
parameterized by depth and widen factor, matching the two configurations used throughout the
paper's CIFAR experiments:

| Factory | Depth | Widen factor | Params (CIFAR-10 head) |
|---|---|---|---|
| `wrn_40_2()` | 40 | 2 | ~2.2M |
| `wrn_28_10()` | 28 | 10 | ~36.5M |

Both are implemented from scratch with no external model-zoo dependency, using
BN→ReLU→Conv pre-activation blocks and a 1×1 shortcut only where channel counts change.

## Extending the augmentation space

Pass any list of `op(img, level)` callables to use a custom space:

```python
from trivial_augment import TrivialAugment, RA_SPACE, rotate, solarize, color

my_space = [rotate, solarize, color]           # a restricted 3-op space
ta = TrivialAugment(augmentation_space=my_space)

wide_space = RA_SPACE + [my_custom_op]          # extend the default space
ta = TrivialAugment(augmentation_space=wide_space, num_strengths=31)
```

An op just needs to accept a `PIL.Image` and an `int` level in `[0, num_strengths)` and return
a `PIL.Image`. This is intentionally the same shape as every function already in
`RA_SPACE`, so new ops compose immediately — see [The impact of randomly pruned spaces](#ablations-worth-knowing)
below before assuming more operations is strictly better.

## Reference results

Top-1 test accuracy (%) as reported in the paper (Table 2), TA using the **Wide** augmentation
space unless noted:

| Model | Dataset | No augmentation | TrivialAugment |
|---|---|---|---|
| Wide-ResNet-40-2 | CIFAR-10 | 96.16 | 96.32 |
| Wide-ResNet-28-10 | CIFAR-10 | 97.03 | 97.46 |
| ShakeShake-26-2x96d | CIFAR-10 | 97.54 | 98.21 |
| PyramidNet | CIFAR-10 | 97.95 | 98.58 |
| Wide-ResNet-40-2 | CIFAR-100 | 78.42 | 79.86 |
| Wide-ResNet-28-10 | CIFAR-100 | 82.22 | 84.33 |
| ShakeShake-26-2x96d | CIFAR-100 | 83.28 | 86.19 |
| Wide-ResNet-28-10 | SVHN Core | 97.12 | 98.11 |
| Wide-ResNet-28-10 | SVHN (full) | 98.67 | 98.9 |
| ResNet-50 | ImageNet | 77.20 | 78.07 |

For context, on the same benchmarks the paper reports AutoAugment/RandAugment/Fast AutoAugment
land within ~0.1–0.5 points of TA almost everywhere, despite costing 1×–800× more search
compute — see the [compute cost comparison](#compute-cost-comparison) below.

This repo reimplements the augmentation and a faithful training recipe, not the exact cluster
setup (epoch counts, batch size, hardware) used in the paper, so treat these as targets to
land close to rather than numbers to match exactly.

## Why it works: the averaging-of-distributions argument

Because TA applies exactly one augmentation per image, sampled uniformly from `|A|` options,
the paper frames the resulting training distribution as an **unweighted average of the `|A|`
data distributions** produced by each augmentation applied to the full dataset on its own —
no combinatorial explosion, no correlated joint distortions from stacking multiple ops. This
is different from RA/AA/UA, which apply `n ≥ 1` chained augmentations and therefore sample
from a distribution over *combinations*. The paper's intuition (Figure 2) is that a simpler,
mixture-of-marginals training distribution is easier for a downstream model to actually learn
from than a combinatorially-compounded one — and that resampling strength per image, not
just per run, is what supplies the useful randomness. Their ablations (Section 4.2) back this
up empirically:

- Randomness in the **strength**, not just the operation, is called out as one of the paper's
  three main takeaways.
- The augmentation space matters more than any tuning of the sampling procedure itself.

## Ablations worth knowing

From the paper's Section 4.2, useful priors if you're adapting this to a new dataset:

- **Bigger augmentation spaces aren't automatically better.** The `Full` space (RA + blur,
  smooth, horizontal/vertical flip) underperforms the smaller, hand-curated spaces on both
  SVHN Core and CIFAR-10.
- **Pruning the space is fairly safe.** Randomly dropping up to 4 of the 14 RA operations
  barely moves accuracy; variance increases as the space shrinks further, but mean
  performance degrades slowly.
- **`invert` is dataset-dependent.** Removing `invert` from the AA space helps CIFAR-10 but
  hurts SVHN Core — consistent with prior literature on invert's effect on digit datasets.
  If you're working with a dataset where global pixel inversion changes semantic meaning
  (e.g. text, digits, sensor readings), consider dropping it explicitly.
- **Strength granularity matters less than you'd think.** Reducing from 31 discrete strengths
  to just 2 or 3 costs almost nothing on CIFAR-10/100/SVHN; using a single fixed strength
  (`{30}`) is the one setting that clearly underperforms on CIFAR-100. A *mixture* of weak and
  strong augmentations, not fine strength granularity, is what matters.

## Compute cost comparison

Search overhead relative to a single training run, as estimated in the paper (Table 1 / Figure
3, RTX 2080 Ti GPU-hours):

| Method | Search overhead |
|---|---|
| AutoAugment | 40×–800× |
| RandAugment | 4×–80× |
| Fast AutoAugment | ~1× |
| **TrivialAugment** | **0×** |

At every fixed compute budget in the paper's cost-vs-accuracy plot (Figure 3), TA sits at or
above the accuracy frontier of every other method, because it spends its entire compute
budget on model training rather than policy search.

## Design decisions in this implementation

- **PIL-based, not tensor-based.** Operations mirror the reference AutoAugment/RandAugment
  codebases (`Pillow`'s `ImageOps`/`ImageEnhance`), applied before `ToTensor()`. This keeps
  the augmentation space directly comparable to published AA/RA/UA implementations, at the
  cost of running on CPU inside the dataloader rather than on-GPU.
  fully deterministic given a fixed seed and simplifies debugging via visual inspection
  (`demo.py`).
- **Direction randomization inline.** Rather than a separate "sign" parameter, each op flips
  its own sign with `random.random() > 0.5`, matching the AA/RA reference behavior instead of
  introducing a new hyperparameter.
- **No `cutout` in the default space.** Cutout is applied as a fixed, separate post-processing
  step in `src/train.py` (matching the paper's CIFAR recipe: TA op, then a final 16px cutout)
  rather than being one of the sampled ops — this matches how the paper's main CIFAR pipeline
  is structured, not the `OHL`/`Full` ablation spaces that fold cutout into `A` itself.

## Known limitations

- The paper explicitly notes TA does **not** work out-of-the-box for object detection; it
  needs further tuning outside the image-classification setting this repo targets.
- `src/model.py` covers WideResNet only. PyramidNet, ShakeShake, and ResNet-50/EfficientNet
  results from the paper are not reimplemented here — plug in your own architecture and reuse
  `src/train.py`'s loop and `trivial_augment.py`'s transform as-is.
- CPU-bound PIL augmentation can bottleneck very high-throughput ImageNet-scale training;
  increase `--workers` or port `trivial_augment.py`'s ops to a tensor-based library (e.g.
  `torchvision.transforms.v2` or `kornia`) if the dataloader becomes the bottleneck.

## FAQ

**Is this the same as RandAugment with `n=1`?**
No. RandAugment fixes a single strength `m` for the entire dataset/run, found via grid search.
TrivialAugment resamples the strength independently for every image, and does no search at
all — that's the actual algorithmic difference, not just a naming one.

**Do I need to tune anything?**
No — that's the point. The only "hyperparameters" are the choice of augmentation space and
`num_strengths`, both of which the paper shows are robust to reasonable choices (see
[Ablations](#ablations-worth-knowing)).

**Can I use this outside image classification?**
The paper only validates classification; it explicitly reports TA needing extra tuning for
object detection. Treat other tasks (segmentation, detection, self-supervised pretraining) as
unvalidated territory.

## Project layout

```
trivialaugment-lite/
├── src/
│   ├── trivial_augment.py   # the augmentation: RA_SPACE ops + TrivialAugment class
│   ├── model.py               # WideResNet-28-10 / WideResNet-40-2, from scratch
│   ├── train.py                # CIFAR-10 / CIFAR-100 training loop, checkpointing
│   └── evaluate.py             # standalone checkpoint evaluation
├── demo.py                     # PIL-only demo, no torch required → augmented_grid.png
├── augmented_grid.png          # sample output of demo.py
├── report.html                  # visual writeup of the method and this implementation
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{muller2021trivialaugment,
  title   = {TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation},
  author  = {M{\"u}ller, Samuel G. and Hutter, Frank},
  journal = {arXiv preprint arXiv:2103.10158},
  year    = {2021}
}
```

## License

MIT — for the code in this repository. The original TrivialAugment paper and its official
code (https://github.com/automl/trivialaugment) are the authors' own work; this is an
independent reimplementation for reference and experimentation.
