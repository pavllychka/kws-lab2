import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data import KWSDataset
from src.model import SmallCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, opt=None):
    training = opt is not None
    model.train(training)
    total = correct = 0
    loss_sum = 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    train_ds = KWSDataset("training", args.classes, args.data_dir)
    val_ds = KWSDataset("validation", args.classes, args.data_dir)
    test_ds = KWSDataset("testing", args.classes, args.data_dir)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, args.batch_size)

    model = SmallCNN(len(args.classes))
    n_params = sum(p.numel() for p in model.parameters())
    print("Параметрів:", n_params)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "model.pt")
    best = 0.0
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, loss_fn, opt)
        vl, va = run_epoch(model, val_dl, loss_fn)
        sched.step()
        print(f"Епоха {ep}/{args.epochs} | train_loss={tl:.4f} "
              f"train_acc={ta * 100:.2f}% | val_loss={vl:.4f} "
              f"val_acc={va * 100:.2f}%")
        if va > best:
            best = va
            torch.save(model.state_dict(), model_path)

    # фінальна оцінка найкращої версії на тестовому наборі
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in DataLoader(test_ds, args.batch_size):
            y_pred += model(x).argmax(1).tolist()
            y_true += y.tolist()
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) * 100
    report = classification_report(
        y_true, y_pred, target_names=args.classes, output_dict=True)

    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(args.classes, f, ensure_ascii=False)
    metrics = {
        "accuracy": round(acc, 2),
        "val_accuracy": round(best * 100, 2),
        "test_samples": len(test_ds),
        "parameters": n_params,
        "model_size_kb": round(os.path.getsize(model_path) / 1024, 1),
        "epochs": args.epochs,
        "seed": args.seed,
        "per_class_f1": {c: round(report[c]["f1-score"], 3)
                         for c in args.classes},
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()