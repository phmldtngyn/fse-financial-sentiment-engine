"""
Orchestrator retrain tự động — cơ chế champion/challenger.

Vòng đời một lần chạy (thích hợp cho cron / GitHub Actions):
    0. Pre-flight  : kiểm tra config, dữ liệu, môi trường, chất lượng split.
                     Lỗi nghiêm trọng → thoát mã 2, KHÔNG train.
    1. Train       : gọi `python -m src.training.train_fse_sft` (3 seeds).
                     Thất bại → thoát mã 1, giữ nguyên champion.
    2. Đối chiếu   : challenger (test F1-macro mới) vs champion
                     (checkpoints/production/manifest.json).
    3. Quyết định  : challenger >= champion + margin → PROMOTE
                     ngược lại → giữ champion, ghi lý do từ chối.
    4. Thoát mã 0  : pipeline chạy trọn vẹn (dù promote hay giữ champion).

Lần chạy đầu tiên chưa có champion → dùng --force để promote ngay.

Cách dùng (từ thư mục gốc repo):
    python -m scripts.retrain_orchestrator --config config/train_config.yaml
    python -m scripts.retrain_orchestrator --dry-run          # chỉ pre-flight
    python -m scripts.retrain_orchestrator --force            # lần đầu tiên
    python -m scripts.retrain_orchestrator --skip-train       # promote thủ công từ báo cáo có sẵn
    python -m scripts.retrain_orchestrator --rollback         # hoàn tác về champion trước

Mã thoát: 0 = OK | 1 = train thất bại | 2 = pre-flight thất bại | 3 = lỗi khác
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    TZ_VN = None

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"

EXIT_OK, EXIT_TRAIN_FAILED, EXIT_PREFLIGHT, EXIT_OTHER = 0, 1, 2, 3


def now_vn() -> str:
    tz = TZ_VN or datetime.now().astimezone().tzinfo
    return datetime.now(tz).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(TZ_VN).strftime("%Y%m%d_%H%M%S") if TZ_VN else datetime.now().strftime("%Y%m%d_%H%M%S")


def git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"orchestrator_{datetime.now(TZ_VN).strftime('%Y%m%d') if TZ_VN else stamp()}.log"
    logger = logging.getLogger("fse.orchestrator")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
    for h in logger.handlers:
        h.setFormatter(fmt)
    logger.info("Log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Bước 0 — Pre-flight
# ---------------------------------------------------------------------------
def preflight(cfg_path: Path, logger):
    """Kiểm tra môi trường + dữ liệu. Trả về (cfg, [cảnh báo]); raise nếu lỗi chết."""
    errors: list[str] = []
    warnings: list[str] = []

    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {cfg_path}")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # torch + GPU
    try:
        import torch

        logger.info("torch %s | CUDA khả dụng: %s", torch.__version__, torch.cuda.is_available())
    except ImportError:
        raise RuntimeError("Chưa cài torch — hãy cài requirements.txt gốc của repo.")

    # underthesea (chỉ bắt buộc khi data chưa tách từ — ở đây chỉ cảnh báo)
    if cfg.get("segmentation", {}).get("mode", "auto") in ("auto", "on"):
        try:
            import underthesea  # noqa: F401
        except ImportError:
            warnings.append("underthesea chưa cài — nếu dữ liệu chưa tách từ thì training sẽ fail "
                            "(pip install underthesea).")

    # mlflow (tùy chọn)
    if cfg.get("mlflow", {}).get("enabled", True):
        try:
            import mlflow  # noqa: F401
        except ImportError:
            warnings.append("mlflow chưa cài → lần train này sẽ không có tracking.")

    # dữ liệu
    d_cfg = cfg["data"]
    file_map = {"train": d_cfg["train_file"], "val": d_cfg["val_file"], "test": d_cfg["test_file"]}
    paths = {}
    sys_path = [str(REPO_ROOT)]
    for name, fname in file_map.items():
        p = REPO_ROOT / d_cfg["processed_dir"] / fname
        if not p.exists():
            errors.append(f"Thiếu file dữ liệu {name}: {p}")
        paths[name] = p
    if errors:
        raise FileNotFoundError("; ".join(errors))

    # chất lượng split
    sys.path[:0] = sys_path
    try:
        from src.data.dataset import load_parquet, verify_splits

        dfs = {
            name: load_parquet(p, d_cfg["text_col"], d_cfg["label_col"], d_cfg.get("min_text_chars", 0))
            for name, p in paths.items()
        }
        for w in verify_splits(dfs["train"], dfs["val"], dfs["test"],
                               text_col=d_cfg["text_col"], label_col=d_cfg["label_col"],
                               date_col=d_cfg.get("date_col")):
            warnings.append(f"SPLIT: {w}")
    except Exception as exc:  # không chặn pipeline nếu lỗi đọc dữ liệu sẽ được bắt lại khi train
        warnings.append(f"Không kiểm tra được split (lý do: {exc})")

    # thư mục output ghi được
    for d in (cfg["output"]["checkpoint_dir"], cfg.get("output", {}).get("report_dir", "reports/training"),
              cfg.get("promote", {}).get("production_dir", "checkpoints/production")):
        try:
            (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"Không tạo được thư mục '{d}': {exc}")
    if errors:
        raise OSError("; ".join(errors))

    return cfg, warnings


# ---------------------------------------------------------------------------
# Bước 1 — Train (subprocess, cách ly môi trường)
# ---------------------------------------------------------------------------
def run_training(cfg_path: Path, logger, extra_args: list[str] | None = None) -> int:
    cmd = [sys.executable, "-m", "src.training.train_fse_sft", "--config", str(cfg_path)]
    if extra_args:
        cmd += extra_args
    logger.info("TRAIN: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


# ---------------------------------------------------------------------------
# Bước 2-3 — Champion/challenger + promote
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def promote(cfg, challenger_report, logger, resolved_value=None) -> dict:
    """Thăng chức challenger → production, xoay vòng history để rollback."""
    p_cfg = cfg.get("promote", {})
    prod_dir = REPO_ROOT / p_cfg.get("production_dir", "checkpoints/production")
    selected_dir = REPO_ROOT / cfg["output"]["checkpoint_dir"] / "selected"
    hist_dir = prod_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    champion_manifest = load_json(prod_dir / "manifest.json")
    summary = load_json(selected_dir.parent / "training_summary.json") or {}

    # lưu champion cũ vào history trước khi ghi đè
    if champion_manifest and (prod_dir / "best.pt").exists():
        ts = champion_manifest.get("promoted_at", "old").replace(":", "").replace("-", "").replace("+", "")
        shutil.copy2(prod_dir / "best.pt", hist_dir / f"{ts}_best.pt")
        shutil.copy2(prod_dir / "manifest.json", hist_dir / f"{ts}_manifest.json")

    shutil.copy2(selected_dir / "best.pt", prod_dir / "best.pt")

    manifest = {
        "promoted_at": now_vn(),
        "first_promotion": champion_manifest is None,
        "metric": p_cfg.get("metric", "test_f1_macro"),
        "value": resolved_value if resolved_value is not None else challenger_report.get("f1_macro"),
        "previous_champion": (
            {"value": champion_manifest.get("value"), "promoted_at": champion_manifest.get("promoted_at")}
            if champion_manifest else None
        ),
        "best_seed": challenger_report.get("best_seed"),
        "backbone": (challenger_report.get("meta") or {}).get("backbone"),
        "data_hash": (challenger_report.get("meta") or {}).get("data_hash"),
        "git_hash": git_hash(),
        "val_f1_mean": summary.get("val_f1_mean"),
        "val_f1_std": summary.get("val_f1_std"),
        "source_report": str(selected_dir / "eval_report.json"),
        "source_checkpoint": str(selected_dir / "best.pt"),
    }
    (prod_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")

    # xoay vòng history, giữ N bản mới nhất
    keep = int(p_cfg.get("keep_versions", 3))
    versioned = sorted(hist_dir.glob("*_best.pt"))
    for old in versioned[:-keep]:
        old.unlink(missing_ok=True)
        old.with_name(old.name.replace("_best.pt", "_manifest.json")).unlink(missing_ok=True)

    logger.info("PROMOTE ✓ challenger (%s=%.4f) → %s",
                manifest["metric"], manifest["value"], prod_dir / "best.pt")
    return manifest


def reject(cfg, challenger_report, champion_manifest, logger, resolved_value=None) -> dict:
    p_cfg = cfg.get("promote", {})
    metric = p_cfg.get("metric", "test_f1_macro")
    margin = float(p_cfg.get("margin", 0.01))
    ch_val = resolved_value if resolved_value is not None else challenger_report.get("f1_macro")
    cp_val = champion_manifest.get("value")
    record = {
        "rejected_at": now_vn(),
        "reason": f"{metric}: challenger {ch_val} < champion {cp_val} + margin {margin}",
        "challenger": {"value": ch_val, "best_seed": challenger_report.get("best_seed"),
                       "report": challenger_report.get("checkpoint")},
        "champion": {"value": cp_val, "promoted_at": champion_manifest.get("promoted_at")},
        "margin": margin,
    }
    out = REPO_ROOT / cfg.get("output", {}).get("report_dir", "reports/training")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"challenger_rejected_{stamp()}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), "utf-8")
    logger.info("GIỮ CHAMPION — %s (báo cáo: %s)", record["reason"], path)
    return record


def rollback(cfg, logger) -> int:
    prod_dir = REPO_ROOT / cfg.get("promote", {}).get("production_dir", "checkpoints/production")
    hist_dir = prod_dir / "history"
    versions = sorted(hist_dir.glob("*_best.pt"))
    if not versions:
        logger.error("Không có bản nào trong history để rollback.")
        return EXIT_OTHER
    src = versions[-1]
    shutil.copy2(src, prod_dir / "best.pt")
    m_src = src.with_name(src.name.replace("_best.pt", "_manifest.json"))
    if m_src.exists():
        shutil.copy2(m_src, prod_dir / "manifest.json")
    logger.info("ROLLBACK ✓ khôi phục %s → %s", src.name, prod_dir / "best.pt")
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def resolve_metric(report: dict, metric: str):
    """Tra metric trong báo cáo, chấp nhận tên có tiền tố "test_"/"val_".

    Config đặt promote.metric: test_f1_macro (tên ngữ nghĩa), còn báo cáo
    eval_report.json lưu key "f1_macro" — hàm này bắc cầu giữa hai cách gọi.
    """
    if metric in report:
        return report[metric]
    if metric.startswith(("test_", "val_")):
        return report.get(metric.split("_", 1)[1])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator retrain tự động FSE")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ pre-flight, không train")
    parser.add_argument("--force", action="store_true", help="Promote bỏ qua so sánh margin (lần đầu)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Dùng báo cáo selected/ đã có (promote thủ công, không train lại)")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--rollback", action="store_true", help="Khôi phục champion từ history")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== FSE retrain orchestrator === git=%s | repo=%s", git_hash(), REPO_ROOT)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    try:
        cfg, warnings = preflight(cfg_path, logger)
    except Exception as exc:
        logger.error("PRE-FLIGHT THẤT BẠI: %s", exc)
        return EXIT_PREFLIGHT

    for w in warnings:
        logger.warning("PRE-FLIGHT: %s", w)

    if args.dry_run:
        logger.info("DRY-RUN: pre-flight OK — sẽ train seeds=%s rồi so với champion (margin=%s).",
                    cfg.get("seeds"), cfg.get("promote", {}).get("margin"))
        return EXIT_OK

    if args.rollback:
        return rollback(cfg, logger)

    # ---- Bước 1: train ----
    if not args.skip_train:
        extra = ["--no-mlflow"] if args.no_mlflow else []
        rc = run_training(cfg_path, logger, extra)
        if rc != 0:
            logger.error("TRAIN THẤT BẠI (mã %d) — giữ nguyên champion.", rc)
            return EXIT_TRAIN_FAILED
    else:
        logger.info("SKIP-TRAIN: dùng báo cáo có sẵn trong selected/.")

    # ---- Bước 2: đọc báo cáo challenger ----
    selected_report = REPO_ROOT / cfg["output"]["checkpoint_dir"] / "selected" / "eval_report.json"
    try:
        challenger = load_json(selected_report)
        if challenger is None:
            raise FileNotFoundError(selected_report)
    except Exception as exc:
        logger.error("Không đọc được báo cáo challenger: %s", exc)
        return EXIT_OTHER

    # ---- Bước 3: so với champion và quyết định ----
    p_cfg = cfg.get("promote", {})
    metric = p_cfg.get("metric", "test_f1_macro")
    margin = float(p_cfg.get("margin", 0.01))
    prod_dir = REPO_ROOT / p_cfg.get("production_dir", "checkpoints/production")
    champion_manifest = load_json(prod_dir / "manifest.json")

    ch_val = resolve_metric(challenger, metric)
    if ch_val is None:
        logger.error("Báo cáo challenger thiếu metric '%s' (key khả dụng: %s).",
                    metric, sorted(k for k in challenger if isinstance(challenger[k], (int, float))))
        return EXIT_OTHER

    if args.force or champion_manifest is None:
        manifest = promote(cfg, challenger, logger, resolved_value=ch_val)
        if champion_manifest is None and not args.force:
            logger.info("Lần đầu promote (chưa có champion).")
    else:
        cp_val = champion_manifest.get("value")
        if cp_val is None or ch_val >= float(cp_val) + margin:
            promote(cfg, challenger, logger, resolved_value=ch_val)
        else:
            reject(cfg, challenger, champion_manifest, logger, resolved_value=ch_val)

    logger.info("=== HOÀN TẤT orchestrator (mã thoát 0) ===")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
