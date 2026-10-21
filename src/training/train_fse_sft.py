"""SFT PhoBERT-base-v2 trên gold standard — không API, offline hoàn toàn.

Usage: python -m src.training.train_fse_sft
"""
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.dataset import FSEDataset
from src.models.fse_model import FSEModel
from src.training.losses import FSELoss


def evaluate(model, dl, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            scores = b["score"].to(device)
            tiers = b["tier"].to(device)
            score_pred, tier_logits = model.forward_batch(ids, mask)
            loss, _ = loss_fn(score_pred, tier_logits, scores, tiers)
            total_loss += loss.item()
            total_mae += (score_pred - scores).abs().sum().item()
            n += len(scores)
    return total_loss / len(dl), total_mae / n


def main():
    with open("config/hyperparameters.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["fse"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr = FSEDataset("data/processed/train.parquet", cfg["backbone"], cfg["max_len"])
    va = FSEDataset("data/processed/val.parquet", cfg["backbone"], cfg["max_len"])
    print(f"Train: {len(tr):,} | Val: {len(va):,}")

    tr_dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = FSEModel(cfg["backbone"]).to(device)
    loss_fn = FSELoss(cfg["lambda_mse"], cfg["lambda_ce"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(tr_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        pbar = tqdm(tr_dl, desc=f"Epoch {ep}/{cfg['epochs']}")
        for b in pbar:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            scores = b["score"].to(device)
            tiers = b["tier"].to(device)

            score_pred, tier_logits = model.forward_batch(ids, mask)
            loss, parts = loss_fn(score_pred, tier_logits, scores, tiers)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            ep_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{parts['mse']:.4f}",
                "ce": f"{parts['ce']:.4f}",
            })

        train_avg = ep_loss / len(tr_dl)
        val_loss, val_mae = evaluate(model, va_dl, loss_fn, device)
        print(f"Epoch {ep} | train={train_avg:.4f} | val={val_loss:.4f} | MAE={val_mae:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ✅ saved best (val={val_loss:.4f})")

    print(f"\n✅ Done. Best val loss: {best_val:.4f}")
    print(f"   Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()