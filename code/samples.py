"""Draw sample beams from a few trained runs, next to the reference designs, for a visual check.

Uses the same loading and sampling path as `python -m engiopt.evaluate` (same conditions, same
latent seed), so the images are the designs that were scored.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engiopt.evaluation.evaluator import Evaluator
from engiopt.utils.all_generators import BUILTIN_GENERATORS

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
N = 8

RUNS = [  # (arm, seed, label)
    ("stable_g1e-3_d1e-4", 2, "Generator faster, seed 2\nMMD 0.017"),
    ("default_g1e-4_d4e-4", 3, "Script default, seed 3\nMMD 0.039"),
    ("default_g1e-4_d4e-4", 7, "Script default, seed 7\nMMD 0.107"),
    ("published11_g3e-4_d3e-4", 8, "Published config, seed 8\nMMD 0.259"),
    ("published11_g3e-4_d3e-4", 2, "Published config, seed 2\nMMD 0.556"),
    ("discfast_g1e-4_d1e-3", 8, "Discriminator faster, seed 8\nMMD 0.823"),
    ("default_g1e-4_d4e-4", 9, "Script default, seed 9\nMMD 0.924"),
]

ev = Evaluator.for_problem("beams2d", spec="beams2d/v1")
ref = np.asarray(ev.resolved.ref_designs)
print("reference designs", ref.shape, "min", ref.min(), "max", ref.max(), "mean", round(float(ref.mean()), 3))
rows = [("Reference designs\n(from the dataset)", ref)]


def stats(x: np.ndarray) -> str:
    flat = x.reshape(len(x), -1)
    grey = float(((flat > 0.2) & (flat < 0.8)).mean())  # share of pixels that are neither solid nor empty
    diversity = float(flat.std(axis=0).mean())  # how much samples differ from each other, per pixel
    return f"mean density {flat.mean():.3f} | grey pixels {grey:.2f} | spread between samples {diversity:.3f}"


print("reference:", stats(ref))
cls = BUILTIN_GENERATORS["cgan_cnn_2d"]
for arm, seed, label in RUNS:
    gen = cls.from_pretrained(
        ev.problem, problem_id="beams2d", seed=seed, device=ev.device, model_source="local",
        hf_entity=None, hf_repo_prefix=None,
        local_model_dir=os.path.join(ROOT, "runs", arm, f"seed_{seed}"), config_fingerprint=None,
    )
    designs = np.asarray(ev._sample(gen))
    print(f"{arm} seed {seed}: shape {designs.shape} min {designs.min():.3f} max {designs.max():.3f} | {stats(designs)}")
    rows.append((label, designs))

fig, axes = plt.subplots(len(rows), N + 1, figsize=(2.0 * (N + 1.6), 1.15 * len(rows)),
                         gridspec_kw={"width_ratios": [1.9] + [2] * N})
for r, (label, designs) in enumerate(rows):
    axes[r, 0].axis("off")
    axes[r, 0].text(0.0, 0.5, label, fontsize=8, va="center", ha="left")
    for c in range(N):
        ax = axes[r, c + 1]
        ax.imshow(np.asarray(designs[c]).reshape(ref.shape[1:]), cmap="gray_r", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle("cgan_cnn_2d on beams2d: the first 8 of the 50 scored samples per run (same conditions in each column)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("saved", OUT)
