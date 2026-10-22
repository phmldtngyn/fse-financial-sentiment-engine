"""SFT PhoBERT 3-class sentiment classification — offline, không API.

Usage: python -m src.training.train_fse_sft
"""
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from src.data.dataset import FSEDataset3Class
from src.models.fse_model import FSEModel
from src.training.losses import FSELoss3Class


def evaluate(model, dl, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            labels = b["label"].to(device)

            logits = model.forward_batch(ids, mask)
            loss, _ = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    return total_loss / len(dl), acc, f1, cm


def main():
    with open("config/hyperparameters.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["fse"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr = FSEDataset3Class("data/processed/train_3class.parquet", cfg["backbone"], cfg["max_len"])
    va = FSEDataset3Class("data/processed/val_3class.parquet", cfg["backbone"], cfg["max_len"])
    print(f"Train: {len(tr):,} | Val: {len(va):,}")

    tr_dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = FSEModel(cfg["backbone"]).to(device)
    loss_fn = FSELoss3Class(class_weights=[2.75, 0.50, 1.54])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(tr_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        pbar = tqdm(tr_dl, desc=f"Epoch {ep}/{cfg['epochs']}")
        for b in pbar:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            labels = b["label"].to(device)

            logits = model.forward_batch(ids, mask)
            loss, _ = loss_fn(logits, labels)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            ep_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_avg = ep_loss / len(tr_dl)
        val_loss, val_acc, val_f1, cm = evaluate(model, va_dl, loss_fn, device)

        print(f"\nEpoch {ep} | train={train_avg:.4f} | val={val_loss:.4f} | "
              f"acc={val_acc:.4f} | F1={val_f1:.4f}")
        print(f"  Confusion matrix (rows=true 0/1/2):\n{cm}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ✅ saved best (F1={val_f1:.4f})\n")

    print(f"\n✅ Done. Best F1: {best_f1:.4f}")
    print(f"   Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()