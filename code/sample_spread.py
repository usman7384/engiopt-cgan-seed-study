"""For every trained run, measure how much its 50 scored samples differ from each other.

A generator that has collapsed draws nearly the same image whatever it is given, so the per-pixel
standard deviation across its samples is close to zero. Uses the same loading and sampling path as
`python -m engiopt.evaluate`, so these are the designs that were scored.

Usage: python sample_spread.py <out.csv>
"""
import csv
import os
import sys

import numpy as np

from engiopt.evaluation.evaluator import Evaluator
from engiopt.utils.all_generators import BUILTIN_GENERATORS

ROOT = os.path.dirname(os.path.abspath(__file__))
ARMS = ["default_g1e-4_d4e-4", "published11_g3e-4_d3e-4", "stable_g1e-3_d1e-4", "discfast_g1e-4_d1e-3"]


def spread(x: np.ndarray) -> float:
    return float(x.reshape(len(x), -1).std(axis=0).mean())


ev = Evaluator.for_problem("beams2d", spec="beams2d/v1")
ref = np.asarray(ev.resolved.ref_designs)
cls = BUILTIN_GENERATORS["cgan_cnn_2d"]
rows = [{"arm": "reference", "seed": "", "sample_spread": round(spread(ref), 4), "mean_density": round(float(ref.mean()), 4)}]
for arm in ARMS:
    for seed in range(1, 11):
        gen = cls.from_pretrained(
            ev.problem, problem_id="beams2d", seed=seed, device=ev.device, model_source="local",
            hf_entity=None, hf_repo_prefix=None,
            local_model_dir=os.path.join(ROOT, "runs", arm, f"seed_{seed}"), config_fingerprint=None,
        )
        d = np.asarray(ev._sample(gen))
        rows.append({"arm": arm, "seed": seed, "sample_spread": round(spread(d), 4), "mean_density": round(float(d.mean()), 4)})
        print(arm, seed, rows[-1]["sample_spread"])

with open(sys.argv[1], "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", sys.argv[1])
