import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def per_class_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 8) -> dict[str, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes)) # row: true value, col: predict value
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp -fp -fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "cm": cm}

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 8) -> dict[str, float]:
    c = per_class_counts(y_true, y_pred, num_classes)
    eps = 1e-12
    precision = c["tp"] / (c["tp"] + c["fp"] + eps)
    recall = c["tp"] / (c["tp"] + c["fn"] + eps)
    specificity = c["tn"] / (c["tn"] + c["fp"] + eps)
    return {
        "accuracy": float(c["tp"].sum() / c["cm"].sum()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "specificity": float(specificity.mean()),
    }

def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
