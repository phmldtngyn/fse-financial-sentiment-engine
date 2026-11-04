"""
SFT training cho FSE Financial Sentiment — BẢN CHUẨN HÓA (viết lại toàn bộ).

Các lỗi của bản cũ đã được sửa trong file này:
1. KHÔNG SEED                → set_seed() đầy đủ cho mọi seed trong cấu hình.
2. Class weights hardcode    → tính động từ tập train (losses.compute_class_weights).
3. padding="max_length" cứng → dynamic padding theo từng batch (Collator).
4. Chạy fp32 chậm            → AMP (mixed precision) khi có GPU, CPU tự tắt.
5. Early stopping không có   → patience + min_delta theo val F1-macro.
6. Không chạm test set       → đánh giá test đầy đủ, xuất eval_report.json/.md.
7. Chạy 1 lần duy nhất       → chạy N seeds (mặc định [42, 7, 2026]), chọn model
                               theo val F1, báo cáo mean ± std giữa các seed.
8. print() rời rạc           → logging có cấu trúc, metrics.jsonl từng epoch.
9. Checkpoint "mù"           → best.pt kèm metadata: seed, data_hash, git_hash,
                               config snapshot, metrics — biết model sinh ra từ đâu.
10. Không tracking           → MLflow (parent run + child run mỗi seed), tự tắt
                               nếu mlflow chưa cài (--no-mlflow để tắt chủ động).

Cấu trúc output:
    checkpoints/fse_sft/
        seed42/  best.pt + meta.json + metrics.jsonl
        seed7/   ...
        seed2026/...
        selected/  best.pt (copy của seed tốt nhất) + eval_report.json/.md (test)
        training_summary.json   (kết quả tất cả seed)

Chạy từ thư mục gốc repo:
    python -m src.training.train_fse_sft --config config/train_config.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:  # mlflow là tùy chọn — pipeline không được phép chết vì thiếu tracking
    import mlflow

    MLFLOW_OK = True
except ImportError:
    mlflow = None
    MLFLOW_OK = False

from .evaluate import build_model, build_report, load_checkpoint, predict, save_report
from .losses import CLASS_NAMES, FSELoss3Class
from .seed import set_seed, worker_init_fn

logger = logging.getLogger("fse.train")

try:  # zoneinfo có sẵn từ Python 3.9
    from zoneinfo import ZoneInfo
    TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    TZ_VN = timezone.utc


def now_iso() -> str:
    return datetime.now(TZ_VN).isoformat(timespec="seconds")


def git_hash() -> str | None:
    """Hash commit hiện tại (nếu chạy trong git repo) — ghi vào mọi checkpoint."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("data", "segmentation", "model", "training", "output"):
        if section not in cfg:
            raise KeyError(f"Config thiếu section bắt buộc: '{section}'")
    return cfg


def linear_warmup_schedule(optimizer, warmup_ratio: float, total_steps: int) -> LambdaLR:
    """Linear warmup + linear decay — tự cài để không phụ thuộc phiên bản transformers."""
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        remaining = total_steps - warmup
        return max(0.0, (total_steps - step) / remaining) if remaining > 0 else 0.0

    return LambdaLR(optimizer, fn)


def _forward_logits(model, batch, device):
    """Gọi model và chuẩn hóa output (một số model tùy chỉnh trả tensor trần)."""
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    return out.logits if hasattr(out, "logits") else out


# ---------------------------------------------------------------------------
# Train một seed
# ---------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    cfg: dict,
    tokenizer,
    dfs: dict,
    device: torch.device,
    mlflow_active: bool = False,
    model_builder=None,
    data_hash: dict | None = None,
) -> dict:
    """Huấn luyện một seed, lưu best checkpoint kèm metadata, trả về kết quả."""
    t_cfg = cfg["training"]
    es = t_cfg.get("early_stopping", {})
    patience = int(es.get("patience", 3))
    min_delta = float(es.get("min_delta", 0.0))

    set_seed(seed)

    from ..data.dataset import Collator, FSEDataset3Class

    d_cfg, m_cfg, s_cfg = cfg["data"], cfg["model"], cfg["segmentation"]
    datasets = {
        name: FSEDataset3Class(
            df, tokenizer, max_len=m_cfg["max_len"],
            text_col=d_cfg["text_col"], label_col=d_cfg["label_col"],
            segmentation_mode=s_cfg["mode"], segmentation_cache_dir=s_cfg["cache_dir"],
        )
        for name, df in dfs.items()
    }
    collator = Collator(tokenizer)
    loaders = {
        name: DataLoader(
            ds, batch_size=t_cfg["batch_size"], shuffle=(name == "train"),
            num_workers=t_cfg.get("num_workers", 0), collate_fn=collator,
            worker_init_fn=worker_init_fn,
        )
        for name, ds in datasets.items()
    }

    # --- Loss với class weights ĐỘNG từ tập train ---
    train_labels = dfs["train"][d_cfg["label_col"]].to_numpy()
    criterion = FSELoss3Class(
        num_labels=m_cfg["num_labels"],
        class_weights=t_cfg.get("class_weights", "auto"),
        train_labels=train_labels,
        label_smoothing=t_cfg.get("label_smoothing", 0.0),
        focal=bool(t_cfg.get("focal_loss", False)),
        focal_gamma=t_cfg.get("focal_gamma", 2.0),
    ).to(device)
    logger.info("Seed %d | class weights (auto từ train): %s",
                seed, [round(float(w), 3) for w in criterion.weights.tolist()])

    # --- Model + optimizer + scheduler ---
    model = model_builder(m_cfg) if model_builder else build_model(m_cfg)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(t_cfg["lr"]),
                      weight_decay=float(t_cfg.get("weight_decay", 0.01)))
    total_steps = len(loaders["train"]) * int(t_cfg["epochs"])
    scheduler = linear_warmup_schedule(optimizer, t_cfg.get("warmup_ratio", 0.1), total_steps)

    amp_enabled = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    if not amp_enabled:
        logger.info("AMP tắt (%s) — chạy fp32.", "không có GPU" if device.type != "cuda" else "cấu hình")

    # --- Vòng lặp epoch + early stopping ---
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    best = {"f1": -1.0, "epoch": -1, "state": None, "metrics": None}
    stale_epochs = 0

    for epoch in range(1, int(t_cfg["epochs"]) + 1):
        # ---- train ----
        model.train()
        train_loss, n_steps = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast("cuda") if amp_enabled else contextlib.nullcontext()
            with ctx:
                logits = _forward_logits(model, batch, device)
                loss, _components = criterion(logits, batch["labels"].to(device))
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
                optimizer.step()
            scheduler.step()
            train_loss += float(loss.detach())
            n_steps += 1

        # ---- validate ----
        y_true, y_pred, y_prob = predict(model, loaders["val"], device)
        val_report = build_report(y_true, y_pred, y_prob)
        epoch_metrics = {
            "seed": seed, "epoch": epoch,
            "train_loss": round(train_loss / max(1, n_steps), 5),
            "val_accuracy": val_report["accuracy"],
            "val_f1_macro": val_report["f1_macro"],
            "val_f1_weighted": val_report["f1_weighted"],
        }
        logger.info(
            "Seed %d epoch %d/%d | train_loss=%.4f | val_f1_macro=%.4f | val_acc=%.4f",
            seed, epoch, t_cfg["epochs"], epoch_metrics["train_loss"],
            val_report["f1_macro"], val_report["accuracy"],
        )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
        if mlflow_active:
            mlflow.log_metrics({
                "train_loss": epoch_metrics["train_loss"],
                "val_f1_macro": val_report["f1_macro"],
                "val_accuracy": val_report["accuracy"],
            }, step=epoch)

        # ---- early stopping ----
        if val_report["f1_macro"] > best["f1"] + min_delta:
            best = {
                "f1": val_report["f1_macro"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "metrics": val_report,
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("Seed %d dừng sớm ở epoch %d (F1 không cải thiện sau %d epoch).",
                            seed, epoch, patience)
                break

    model.load_state_dict(best["state"])
    model.to(device)
    model.eval()

    # --- Lưu checkpoint kèm metadata (không còn checkpoint "mù") ---
    meta = {
        "seed": seed,
        "backbone": m_cfg.get("backbone"),
        "factory": m_cfg.get("factory"),
        "num_labels": m_cfg["num_labels"],
        "max_len": m_cfg["max_len"],
        "segmentation_mode": s_cfg["mode"],
        "class_weights": [round(float(w), 4) for w in criterion.weights.tolist()],
        "label_smoothing": t_cfg.get("label_smoothing", 0.0),
        "focal_loss": bool(t_cfg.get("focal_loss", False)),
        "best_epoch": best["epoch"],
        "val_metrics": best["metrics"],
        "data_hash": data_hash or {},
        "git_hash": git_hash(),
        "config_snapshot": {"model": m_cfg, "training": {k: v for k, v in t_cfg.items()}},
        "saved_at": now_iso(),
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_dir / "best.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")

    return {
        "seed": seed,
        "best_epoch": best["epoch"],
        "val_f1_macro": best["f1"],
        "val_accuracy": best["metrics"]["accuracy"] if best["metrics"] else None,
        "checkpoint": str(ckpt_dir / "best.pt"),
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Multi-seed orchestration + test evaluation
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="SFT training chuẩn hóa cho FSE")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--seeds", default=None, help="Ghi đè danh sách seed, vd '42,7'")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-mlflow", action="store_true", help="Tắt MLflow cho lần chạy này")
    parser.add_argument("--tag", default=None, help="Ghi chú gắn vào run (vd 'tháng-11')")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.error("Config lỗi: %s", exc)
        return 2

    # ---- dữ liệu: kiểm tra tồn tại TRƯỚC khi tải model (fail nhanh) ----
    d_cfg, s_cfg, m_cfg, t_cfg, o_cfg = (
        cfg["data"], cfg["segmentation"], cfg["model"], cfg["training"], cfg["output"],
    )
    file_map = {"train": d_cfg["train_file"], "val": d_cfg["val_file"], "test": d_cfg["test_file"]}
    data_paths = {}
    try:
        from ..data.dataset import Collator, FSEDataset3Class, load_parquet, verify_splits

        for name, fname in file_map.items():
            data_paths[name] = Path(d_cfg["processed_dir"]) / fname
            if not data_paths[name].exists():
                raise FileNotFoundError(
                    f"Thiếu {name}: {data_paths[name]} — hãy chạy prepare_3class_dataset.py trước."
                )
        dfs = {
            name: load_parquet(p, d_cfg["text_col"], d_cfg["label_col"], d_cfg.get("min_text_chars", 0))
            for name, p in data_paths.items()
        }
        logger.info("Dữ liệu: train=%d | val=%d | test=%d mẫu.",
                    len(dfs["train"]), len(dfs["val"]), len(dfs["test"]))
    except Exception as exc:
        logger.error("Dữ liệu lỗi: %s", exc)
        return 2

    # ---- kiểm tra chất lượng split (split theo thời gian + leakage) ----
    for w in verify_splits(dfs["train"], dfs["val"], dfs["test"],
                           text_col=d_cfg["text_col"], label_col=d_cfg["label_col"],
                           date_col=d_cfg.get("date_col")):
        logger.warning("SPLIT: %s", w)

    data_hash = {name: sha256_file(p) for name, p in data_paths.items()}

    seeds = (
        [int(s) for s in args.seeds.split(",")] if args.seeds
        else [int(s) for s in cfg.get("seeds", [42, 7, 2026])]
    )

    # ---- device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Thiết bị: %s | seeds: %s | backbone: %s", device, seeds, m_cfg["backbone"])

    # ---- MLflow: parent run + child run mỗi seed ----
    mlflow_active = bool(cfg.get("mlflow", {}).get("enabled", False)) and not args.no_mlflow
    if mlflow_active and not MLFLOW_OK:
        logger.warning("mlflow chưa cài (pip install mlflow) → chạy KHÔNG tracking.")
        mlflow_active = False
    if mlflow_active:
        mlf = cfg["mlflow"]
        mlflow.set_tracking_uri(mlf.get("tracking_uri", "./mlruns"))
        mlflow.set_experiment(mlf.get("experiment_name", "fse_sentiment"))
        run_name = f"{mlf.get('run_name_prefix', 'sft')}_{datetime.now(TZ_VN):%Y%m%d_%H%M%S}"

    parent_ctx = (
        mlflow.start_run(run_name=run_name, tags={
            "tag": args.tag or "", "git_hash": git_hash() or "",
            "backbone": m_cfg["backbone"], "seeds": ",".join(map(str, seeds)),
            **{f"data_hash_{k}": v[:12] for k, v in data_hash.items()},
        })
        if mlflow_active else contextlib.nullcontext()
    )

    try:
        with parent_ctx:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(m_cfg["backbone"])

            results = []
            for seed in seeds:
                seed_ctx = (
                    mlflow.start_run(run_name=f"seed{seed}", nested=True)
                    if mlflow_active else contextlib.nullcontext()
                )
                with seed_ctx:
                    if mlflow_active:
                        mlflow.log_params({
                            "seed": seed, "lr": t_cfg["lr"], "batch_size": t_cfg["batch_size"],
                            "epochs": t_cfg["epochs"], "max_len": m_cfg["max_len"],
                            "focal_loss": t_cfg.get("focal_loss", False),
                            "label_smoothing": t_cfg.get("label_smoothing", 0.0),
                        })
                    results.append(train_one_seed(
                        seed, cfg, tokenizer,
                        dfs={"train": dfs["train"], "val": dfs["val"]},
                        device=device, mlflow_active=mlflow_active, data_hash=data_hash,
                    ))

            # ---- tổng hợp multi-seed ----
            f1s = [r["val_f1_macro"] for r in results]
            mean_f1 = sum(f1s) / len(f1s)
            std_f1 = (sum((x - mean_f1) ** 2 for x in f1s) / len(f1s)) ** 0.5
            best_result = max(results, key=lambda r: r["val_f1_macro"])
            logger.info("Val F1-macro theo seed: %s", {r["seed"]: r["val_f1_macro"] for r in results})
            logger.info("Mean=%.4f ± %.4f | chọn seed %d (val_f1=%.4f, epoch %d)",
                        mean_f1, std_f1, best_result["seed"],
                        best_result["val_f1_macro"], best_result["best_epoch"])

            # ---- đánh giá TEST trên model tốt nhất (nạp lại từ checkpoint để
            #      đồng thời xác minh checkpoint nạp được từ đĩa) ----
            model, meta = load_checkpoint(best_result["checkpoint"], device)
            test_loader = DataLoader(
                FSEDataset3Class(
                    dfs["test"], tokenizer, max_len=m_cfg["max_len"],
                    text_col=d_cfg["text_col"], label_col=d_cfg["label_col"],
                    segmentation_mode=s_cfg["mode"], segmentation_cache_dir=s_cfg["cache_dir"],
                ),
                batch_size=t_cfg["batch_size"], shuffle=False,
                collate_fn=Collator(tokenizer),
            )
            y_true, y_pred, y_prob = predict(model, test_loader, device)
            test_report = build_report(y_true, y_pred, y_prob)
            test_report.update({
                "checkpoint": str(Path(cfg["output"]["checkpoint_dir"]) / "selected" / "best.pt"),
                "best_seed": best_result["seed"],
                "meta": {k: meta.get(k) for k in ("seed", "backbone", "saved_at", "data_hash", "git_hash")},
            })

            # ---- lưu artifacts cuối: selected/ + training_summary.json ----
            ckpt_root = Path(cfg["output"]["checkpoint_dir"])
            selected_dir = ckpt_root / "selected"
            selected_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_result["checkpoint"], selected_dir / "best.pt")
            save_report(
                test_report, selected_dir / "eval_report.json", selected_dir / "eval_report.md",
                title=f"FSE test set — seed {best_result['seed']}",
            )
            summary = {
                "created_at": now_iso(),
                "git_hash": git_hash(),
                "data_hash": data_hash,
                "seeds": seeds,
                "val_f1_by_seed": {str(r["seed"]): r["val_f1_macro"] for r in results},
                "val_f1_mean": round(mean_f1, 4),
                "val_f1_std": round(std_f1, 4),
                "best_seed": best_result["seed"],
                "test": {
                    "accuracy": test_report["accuracy"],
                    "f1_macro": test_report["f1_macro"],
                    "f1_weighted": test_report["f1_weighted"],
                    "n_samples": test_report["n_samples"],
                },
                "report_dir": str(cfg.get("output", {}).get("report_dir", "reports/training")),
                "config": {k: cfg[k] for k in ("data", "segmentation", "model", "training")},
            }
            (ckpt_root / "training_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), "utf-8"
            )
            if mlflow_active:
                mlflow.log_metrics({
                    "test_f1_macro": test_report["f1_macro"],
                    "test_accuracy": test_report["accuracy"],
                    "val_f1_mean": mean_f1, "val_f1_std": std_f1,
                })
                mlflow.log_artifacts(str(selected_dir), artifact_path="selected")

            logger.info(
                "HOÀN TẤT | test_f1_macro=%.4f | test_acc=%.4f | selected → %s",
                test_report["f1_macro"], test_report["accuracy"], selected_dir / "best.pt",
            )
            return 0
    except Exception:
        logger.exception("Training thất bại.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
