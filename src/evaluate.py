import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.data import KWSDataset
from src.model import SmallCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    model_path = os.path.join(args.artifacts, "model.pt")
    with open(os.path.join(args.artifacts, "labels.json"), encoding="utf-8") as f:
        classes = json.load(f)

    model = SmallCNN(len(classes))
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    ds = KWSDataset("testing", classes, args.data_dir)
    correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(ds, 64):
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    print(f"Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")
    print(f"Model size: {os.path.getsize(model_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()