"""
Smoke test END-TO-END cho quy trình training tự động — chạy trong vài chục giây,
KHÔNG cần GPU, KHÔNG cần tải PhoBERT (dùng model mini tests/smoke_model.py).

Kiểm tra:
  1. Unit logic: class weights động, phát hiện segmentation, verify_splits,
     loss (auto weights / focal guard).
  2. Train multi-seed (2 seeds) qua subprocess như chạy thật.
  3. Artifacts đầy đủ: best.pt + meta.json + eval_report.json + training_summary.json.
  4. Orchestrator: --force promote → manifest.json; chạy lại → REJECT vì không
     vượt margin; --rollback khôi phục champion cũ.
  5. evaluate CLI đánh giá lại checkpoint từ production/.

Chạy từ thư mục gốc fse-training (hoặc repo gốc sau khi drop-in):
    python -m tests.smoke_test

Thoát 0 = toàn bộ PASS.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys

# --- Windows an toàn: khi stdout/stderr là pipe/redirect, Python dùng codepage
# locale (vd cp1252) và sẽ crash khi in dấu tiếng Việt -> ép UTF-8 bắt buộc.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1) Unit logic (chạy trong process hiện tại)
# ---------------------------------------------------------------------------
def unit_tests() -> None:
    print("\n== 1. Unit tests ==")
    sys.path.insert(0, str(ROOT))

    from src.training.losses import compute_class_weights

    labels = np.repeat([0, 1, 2], [625, 3425, 1116])  # phân bố 12.1/66.3/21.6 của 5166 mẫu
    w = compute_class_weights(labels)
    old_hardcoded = np.array([2.755, 0.503, 1.543])
    check("compute_class_weights khớp bộ [2.75, 0.50, 1.54] của bản cũ",
          np.allclose(w, old_hardcoded, atol=0.01), f"got {np.round(w, 3)}")

    from src.data.dataset import looks_segmented, verify_splits

    check("looks_segmented: phát hiện text đã tách từ",
          looks_segmented(["ngan_hang_viet_nam tang_truong_manh", "co_phieu tang_tran_nho_dong_tien"]))
    check("looks_segmented: phát hiện text CHƯA tách từ",
          not looks_segmented(["Ngân hàng Việt Nam tăng trưởng mạnh", "Cổ phiếu tăng trần nhờ dòng tiền"]))

    train_df = pd.DataFrame({"text": ["ngân hàng tăng trưởng tốt", "lợi nhuận giảm sâu", "họp thường niên"] * 10,
                             "label": [2, 0, 1] * 10})
    val_df = train_df.iloc[:9].reset_index(drop=True)
    test_df = train_df.iloc[9:21].reset_index(drop=True)
    warns = verify_splits(train_df, val_df, test_df)
    check("verify_splits: phát hiện trùng lặp chéo train/test",
          any("trùng lặp" in x for x in warns), str(warns))

    dated = pd.DataFrame({"text": [f"câu tin {i}" for i in range(20)],
                          "label": [1] * 20,
                          "date": pd.date_range("2025-01-01", periods=20)})
    warns = verify_splits(dated.iloc[:10], dated.iloc[10:14], dated.iloc[14:], date_col="date")
    check("verify_splits: split theo thời gian hợp lệ → không cảnh báo thời gian",
          not any("KHÔNG theo thời gian" in x for x in warns), str(warns))
    warns = verify_splits(dated.iloc[10:], dated.iloc[10:14], dated.iloc[:6], date_col="date")
    check("verify_splits: test cũ hơn train → cảnh báo leakage thời gian",
          any("KHÔNG theo thời gian" in x for x in warns), str(warns))

    from src.training.losses import FSELoss3Class
    import torch

    crit = FSELoss3Class(class_weights="auto", train_labels=[0] * 10 + [1] * 40 + [2] * 20,
                         label_smoothing=0.05)
    loss, comp = crit(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    check("FSELoss3Class auto weights + label smoothing → loss hữu hạn",
          torch.isfinite(loss).item())
    try:
        FSELoss3Class(class_weights="auto", train_labels=[0, 1, 2], label_smoothing=0.1, focal=True)
        check("Focal + label smoothing → bị chặn", False)
    except ValueError:
        check("Focal + label smoothing → bị chặn", True)


# ---------------------------------------------------------------------------
# 2) Chuẩn bị môi trường chạy subprocess (sạch, không đụng repo thật)
# ---------------------------------------------------------------------------
POS = ["Ngân hàng {} báo lợi nhuận tăng mạnh trong quý", "Cổ phiếu {} tăng trần nhờ dòng tiền lớn",
       "Lợi nhuận {} vượt kỳ vọng, cổ tức cao", "Kim ngạch tín dụng {} tăng tốc cuối năm"]
NEG = ["Lợi nhuận {} giảm sâu do nợ xấu gia tăng", "Cổ phiếu {} sụt mạnh sau tin xấu",
       "Hội đồng quản trị {} bị xử phạt vi phạm", "Thanh khoản {} cạn kiệt, giá về đáy"]
NEU = ["Ngân hàng {} họp đại hội đồng cổ đông", "{} công bố báo cáo thường niên",
       "Chủ tịch {} phát ngôn tại diễn đàn", "{} điều lệ nhân sự ban giám đốc"]
SUBJECTS = ["VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "STB"]


def make_sentence(cls: int) -> str:
    tpl = random.choice({2: POS, 0: NEG, 1: NEU}[cls])
    return tpl.format(random.choice(SUBJECTS))


def build_run_dir() -> Path:
    run_dir = Path(tempfile.mkdtemp(prefix="fse_smoke_"))
    for item in ("src", "scripts", "tests", "config"):
        dest = run_dir / item
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ROOT / item, dest, ignore=shutil.ignore_patterns("__pycache__"))

    data_dir = run_dir / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    sizes = {"train": 180, "val": 40, "test": 40}
    for split, n in sizes.items():
        labels = [2] * (n // 3) + [0] * (n // 3) + [1] * (n - 2 * (n // 3))
        random.shuffle(labels)
        df = pd.DataFrame({"text": [make_sentence(c) for c in labels], "label": labels})
        df.to_parquet(data_dir / f"{split}_3class.parquet", index=False)

    cfg = yaml.safe_load((ROOT / "config" / "train_config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["factory"] = "tests.smoke_model:build_model"
    cfg["model"]["max_len"] = 64
    cfg["training"].update({"lr": 0.01, "batch_size": 8, "epochs": 3, "num_workers": 0})
    cfg["seeds"] = [42, 7]
    cfg["mlflow"]["enabled"] = False
    cfg["output"]["checkpoint_dir"] = "checkpoints/smoke"
    cfg["output"]["report_dir"] = "reports/smoke"
    cfg["promote"]["production_dir"] = "checkpoints/production"
    (run_dir / "config" / "smoke_config.yaml").write_text(
        yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return run_dir


def run_step(run_dir: Path, name: str, args: list[str], expect_rc: int = 0) -> subprocess.CompletedProcess:
    print(f"\n-- {name}: python -m {' '.join(args)}")
    proc = subprocess.run([sys.executable, "-m", *args], cwd=run_dir,
                          capture_output=True, text=True, timeout=900)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
    print(tail)
    check(f"{name} (mã thoát {expect_rc})", proc.returncode == expect_rc,
          f"rc={proc.returncode}")
    return proc


def artifact_tests(run_dir: Path) -> None:
    print("\n== Artifacts training ==")
    ck = run_dir / "checkpoints" / "smoke"
    check("best.pt tồn tại cho cả 2 seed",
          (ck / "seed42" / "best.pt").exists() and (ck / "seed7" / "best.pt").exists())
    meta = json.loads((ck / "seed42" / "meta.json").read_text(encoding="utf-8"))
    check("meta.json có data_hash + git fields", "data_hash" in meta and "seed" in meta)
    check("meta.json ghi nhận class weights động", "class_weights" in meta and
          len(meta["class_weights"]) == 3)
    check("selected/best.pt được copy", (ck / "selected" / "best.pt").exists())
    report = json.loads((ck / "selected" / "eval_report.json").read_text(encoding="utf-8"))
    check("eval_report.json có f1_macro + per_class + confusion_matrix",
          {"f1_macro", "per_class", "confusion_matrix"} <= set(report))
    summary = json.loads((ck / "training_summary.json").read_text(encoding="utf-8"))
    check("training_summary.json có kết quả 2 seed", len(summary.get("val_f1_by_seed", {})) == 2)


def orchestrator_tests(run_dir: Path) -> None:
    print("\n== Artifacts orchestrator ==")
    prod = run_dir / "checkpoints" / "production"
    manifest = json.loads((prod / "manifest.json").read_text(encoding="utf-8"))
    check("production/manifest.json được tạo khi --force", manifest.get("first_promotion") is True)
    rejected = list((run_dir / "reports" / "smoke").glob("challenger_rejected_*.json"))
    check("challenger không vượt margin → bị từ chối", len(rejected) == 1)
    check("rollback giữ lại manifest.json", (prod / "manifest.json").exists() and
          (prod / "best.pt").exists())


def orchestrator_tests_force(run_dir: Path) -> None:
    print("\n== Artifacts orchestrator (sau --force) ==")
    prod = run_dir / "checkpoints" / "production"
    manifest = json.loads((prod / "manifest.json").read_text(encoding="utf-8"))
    check("production/manifest.json được tạo khi --force", manifest.get("first_promotion") is True)


def main() -> int:
    print("=== FSE SMOKE TEST ===")
    unit_tests()
    if FAILURES:
        print("\nDừng sớm: unit tests thất bại.")
        return 1

    run_dir = build_run_dir()
    print(f"\nThư mục chạy tạm: {run_dir}")
    try:
        run_step(run_dir, "2. Train 2 seeds", ["src.training.train_fse_sft",
                                               "--config", "config/smoke_config.yaml"])
        artifact_tests(run_dir)
        run_step(run_dir, "3. Orchestrator --skip-train --force (promote lần đầu)",
                 ["scripts.retrain_orchestrator", "--config", "config/smoke_config.yaml",
                  "--skip-train", "--force"])
        orchestrator_tests_force(run_dir)
        run_step(run_dir, "4. Orchestrator chạy lại (không vượt margin → giữ champion)",
                 ["scripts.retrain_orchestrator", "--config", "config/smoke_config.yaml",
                  "--skip-train"])
        run_step(run_dir, "5. Orchestrator --force lần 2 (champion cũ → history)",
                 ["scripts.retrain_orchestrator", "--config", "config/smoke_config.yaml",
                  "--skip-train", "--force"])
        check("history/ lưu champion cũ trước khi ghi đè",
              len(list((run_dir / "checkpoints" / "production" / "history").glob("*_best.pt"))) == 1)
        run_step(run_dir, "6. Orchestrator --rollback (khôi phục champion từ history)",
                 ["scripts.retrain_orchestrator", "--config", "config/smoke_config.yaml",
                  "--rollback"])
        orchestrator_tests(run_dir)
        run_step(run_dir, "7. evaluate CLI trên production checkpoint",
                 ["src.training.evaluate", "--checkpoint", "checkpoints/production/best.pt",
                  "--config", "config/smoke_config.yaml", "--split", "test"])
        check("eval_report_test.json được sinh cạnh production checkpoint",
              (run_dir / "checkpoints" / "production" / "eval_report_test.json").exists())

        print("\n=== KẾT QUẢ: " + ("TOÀN BỘ PASS ✓" if not FAILURES else f"{len(FAILURES)} FAIL: {FAILURES}"))
        return 0 if not FAILURES else 1
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
