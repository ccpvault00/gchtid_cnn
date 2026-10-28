"""Generate experiment configs.

stage1: architecture comparison (one knob at a time) under the baseline hyperparameters.
stage2: for default / best custom / best VGG (chosen by val_accuracy in results.csv),
        one hyperparameter changed at a time.
"""
import argparse
import ast
from pathlib import Path

import pandas as pd
import yaml

CFG_DIR = Path("configs")
BASE = {"epochs": 30, "batch_size": 64, "lr": 0.001, "optimizer": "Adam"}
CUSTOM_BASE = {"n_blocks": 4, "base_filters": 32, "kernel_size": 3, "dropout": 0.3}


def write(name: str, cfg: dict) -> None:
    (CFG_DIR / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def stage1() -> None:
    write("s1_default", {"model": "default", "model_kwargs": {"dropout": 0.3}, **BASE})
    write("s1_custom_base", {"model": "custom", "model_kwargs": dict(CUSTOM_BASE), **BASE})
    for b in (2, 6):
        write(f"s1_custom_b{b}", {"model": "custom", "model_kwargs": {**CUSTOM_BASE, "n_blocks": b}, **BASE})
    for k in (5, 7):
        write(f"s1_custom_k{k}", {"model": "custom", "model_kwargs": {**CUSTOM_BASE, "kernel_size": k}, **BASE})
    for f in (64, 128):
        write(f"s1_custom_f{f}", {"model": "custom", "model_kwargs": {**CUSTOM_BASE, "base_filters": f}, **BASE})
    for t in ("all", "none", "last_layer", "last_block"):
        write(f"s1_vgg_{t}", {"model": "vgg16", "model_kwargs": {"trainable": t, "dropout": 0.3}, **BASE})


def stage2() -> None:
    df = pd.read_csv("results.csv")
    df = df[df["run"].str.startswith("s1_")]
    picks = {"default": "s1_default"}
    for tag, prefix in (("custom", "s1_custom_"), ("vgg", "s1_vgg_")):
        sub = df[df["run"].str.startswith(prefix)]
        picks[tag] = sub.loc[sub["val_accuracy"].idxmax(), "run"]
    print("stage2 bases:", picks)
    for tag, run in picks.items():
        src = yaml.safe_load((CFG_DIR / f"{run}.yaml").read_text())
        mk = src["model_kwargs"]
        variants = {
            "ep10": {"epochs": 10}, "ep50": {"epochs": 50},
            "bs32": {"batch_size": 32}, "bs128": {"batch_size": 128},
            "lr0.1": {"lr": 0.1}, "lr0.01": {"lr": 0.01}, "lr0.0001": {"lr": 0.0001},
            "do0.1": {"model_kwargs": {**mk, "dropout": 0.1}}, "do0.5": {"model_kwargs": {**mk, "dropout": 0.5}},
            "adamw": {"optimizer": "AdamW"}, "sgd": {"optimizer": "SGD", "optimizer_kwargs": {"momentum": 0.9}},
            "bce": {"loss": "bce"},
        }
        for vname, change in variants.items():
            write(f"s2_{tag}_{vname}", {**src, **change})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["stage1", "stage2"])
    globals()[ap.parse_args().stage]()
