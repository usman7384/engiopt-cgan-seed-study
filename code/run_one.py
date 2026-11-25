"""Train one cgan_cnn_2d run on beams2d with the unmodified EngiOpt script and save the trained generator to disk.

The EngiOpt training script only saves by uploading to Hugging Face. This wrapper runs it unchanged
(no tracking, no upload), then writes generator.pth + run_config.json + metadata.json into a
folder that `python -m engiopt.evaluate --model-source local` can score.

Usage:
  python run_one.py --arm NAME --seed S --lr-gen X --lr-disc Y --n-epochs E --batch-size B --latent-dim L
"""
import argparse
import json
import os
import runpy
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, "EngiOpt", "engiopt", "generators", "cgan_cnn_2d", "cgan_cnn_2d.py")

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--lr-gen", type=float, required=True)
p.add_argument("--lr-disc", type=float, required=True)
p.add_argument("--n-epochs", type=int, required=True)
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--latent-dim", type=int, required=True)
a = p.parse_args()

pkg = os.path.join(ROOT, "runs", a.arm, f"seed_{a.seed}")
done = os.path.join(pkg, "run_config.json")
if os.path.exists(done):
    print("already done:", pkg)
    sys.exit(0)
os.makedirs(pkg, exist_ok=True)
os.chdir(pkg)  # the training script writes sample images relative to the working directory
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")

sys.argv = [
    SCRIPT, "--problem-id", "beams2d", "--no-track", "--no-save-model",
    "--seed", str(a.seed), "--lr-gen", str(a.lr_gen), "--lr-disc", str(a.lr_disc),
    "--n-epochs", str(a.n_epochs), "--batch-size", str(a.batch_size), "--latent-dim", str(a.latent_dim),
]
t0 = time.time()
g = runpy.run_path(SCRIPT, run_name="__main__")
train_seconds = time.time() - t0

import torch  # noqa: E402  (imported after training so the script controls seeding first)

args = g["args"]
torch.save({"generator": g["generator"].state_dict()}, os.path.join(pkg, "generator.pth"))
cond_keys = list(g["cond_keys"]) if "cond_keys" in g else None
json.dump(
    {
        "algo": "cgan_cnn_2d", "problem_id": "beams2d", "seed": a.seed, "checkpoint_backend": "none",
        "checkpoint_files": ["generator.pth"], "primary_files": ["generator.pth"], "condition_keys": cond_keys,
        "arm": a.arm, "train_seconds": round(train_seconds, 1), "device": str(g.get("device")),
        "torch": torch.__version__,
    },
    open(os.path.join(pkg, "metadata.json"), "w"), indent=1,
)
# run_config.json is written last: its presence marks the run as complete
json.dump({k: v for k, v in vars(args).items()}, open(done, "w"), indent=1, default=str)
print(f"DONE arm={a.arm} seed={a.seed} train_seconds={train_seconds:.1f} pkg={pkg}")
