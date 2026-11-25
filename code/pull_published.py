"""Pull the published metrics and run configs for the EngiOpt cGAN baselines on beams2d.

Reads only public files from the Hugging Face model repo IDEALLab/engiopt-cgan-cnn-2d.
Usage: python pull_published.py <out.json>
"""
import json
import sys
import urllib.request

REPO = "IDEALLab/engiopt-cgan-cnn-2d"
API = f"https://huggingface.co/api/models/{REPO}/tree/main/"
RAW = f"https://huggingface.co/{REPO}/resolve/main/"
HEADERS = {"User-Agent": "python-urllib"}


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8")


def tree(path):
    return json.loads(get(API + path))


rows = []
cfgs = [x["path"] for x in tree("beams2d") if x["type"] == "directory"]
for cfg in cfgs:
    for seed in [x["path"] for x in tree(cfg) if x["type"] == "directory"]:
        row = {"cfg": cfg.split("/")[-1], "seed": seed.split("/")[-1]}
        for name in ("run_config.json", "metrics.json", "metadata.json"):
            try:
                row[name.split(".")[0]] = json.loads(get(RAW + seed + "/" + name))
            except Exception as exc:  # noqa: BLE001
                row[name.split(".")[0]] = {"_error": str(exc)[:80]}
        rows.append(row)
        print(row["cfg"], row["seed"], "ok", flush=True)

json.dump(rows, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("packages:", len(rows), "configs:", len(cfgs))
