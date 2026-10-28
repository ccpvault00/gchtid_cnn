import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
import torch.nn.functional as F

from data import build_loaders, CLASSES
from models import build_model
from metrics import compute_metrics, per_class_counts, plot_confusion_matrix

from tqdm import tqdm

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1004)
    return p.parse_args()


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            target = F.one_hot(y, logits.shape[1]).float() if isinstance(criterion, nn.BCEWithLogitsLoss) else y
            loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / len(loader.dataset)

def build_criterion(name: str) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"unknown loss: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)

def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_seed(args.seed)
    device = torch.device("cuda")

    run_dir = Path("runs") / args.config.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    split_csv = Path("split.csv")
    if not split_csv.exists():
        raise FileNotFoundError("split.csv not found; run: python data.py --root <all_image dir>")
    
    train_loader, val_loader, test_loader = build_loaders(split_csv, cfg["batch_size"])
    model = build_model(cfg["model"], **cfg.get("model_kwargs", {})).to(device)
    criterion = build_criterion(cfg.get("loss", "ce")) 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, cfg["optimizer"])(params, lr=cfg["lr"], **cfg.get("optimizer_kwargs", {}))

    best_acc, t0 = 0.0, time.perf_counter()
    for epoch in tqdm(range(cfg["epochs"]), desc="epochs"):
        te = time.perf_counter()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        val = compute_metrics(y_true, y_pred)
        tqdm.write(f"epoch {epoch+1:02d} loss {loss:.4f} val_acc {val['accuracy']:.4f} ({time.perf_counter()-te:.0f}s)")
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            torch.save(model.state_dict(), run_dir / "best.pt")
    train_time = time.perf_counter() - t0

    # test
    model.load_state_dict(torch.load(run_dir / "best.pt"))
    t1 = time.perf_counter()
    y_true, y_pred = evaluate(model, test_loader, device)
    infer_ms = (time.perf_counter() - t1) / len(test_loader.dataset) * 1000
    test = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(per_class_counts(y_true, y_pred)["cm"], CLASSES, run_dir / "confusion_matrix.png")

    row = {
        "run": args.config.stem, "model": cfg["model"], "model_kwargs": str(cfg.get("model_kwargs", {})),
        "val_accuracy": round(best_acc, 4), **test,
        "train_time_s": round(train_time, 1), "infer_ms_per_img": round(infer_ms, 3),
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "epochs": cfg["epochs"], "batch_size": cfg["batch_size"], "lr": cfg["lr"], "optimizer": cfg["optimizer"],
        "dropout": cfg.get("model_kwargs", {}).get("dropout"), "loss": cfg.get("loss", "ce")
    }


    out = Path("results.csv")
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(row)

if __name__ == "__main__":
    main()
    






