from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torchvision.transforms import v2

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import hashlib

CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def make_split(root: Path, out_csv: Path, seed: int = 1004) -> pd.DataFrame:
    """
    Scan the ImageFolder, make 70/15/15 stratified split and save as CSV.
    """
    rows = []
    for cls in CLASSES:
        for p in sorted((root / cls).glob("*.png")):
            rows.append({"path": str(p), "label": CLASSES.index(cls)})
    df = pd.DataFrame(rows)

    df["md5"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in df["path"]]
    df = df.drop_duplicates("md5").drop(columns="md5").reset_index(drop=True)

    train, tmp = train_test_split(df, test_size= 0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(tmp, test_size= 0.50, stratify=tmp["label"], random_state=seed)
    train["split"], val["split"], test["split"] = "train", "val", "test"

    df = pd.concat([train, val, test]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    
    return df

def build_transforms(train: bool) -> v2.Compose:
    ops = []
    if train:
        ops += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
    ops += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    return v2.Compose(ops)

class GCHTIDDataset(Dataset):
    def __init__(self, csv_path: Path, split: str, transform: v2.Compose) -> None:
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])

def build_loaders(csv_path: Path, batch_size: int, num_workers: int = 8) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train = DataLoader(GCHTIDDataset(csv_path, "train", build_transforms(True)), shuffle=True, drop_last=True, **common)
    val = DataLoader(GCHTIDDataset(csv_path, "val", build_transforms(False)), shuffle=False, **common)
    test = DataLoader(GCHTIDDataset(csv_path, "test", build_transforms(False)), shuffle=False, **common)
    return train, val, test

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("split.csv"))
    p.add_argument("--seed", type=int, default=1004)
    
    a=p.parse_args()
    df = make_split(a.root, a.out, a.seed)
    print(df.groupby(["split", "label"]).size().unstack())
