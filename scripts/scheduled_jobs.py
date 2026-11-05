"""
Scheduler trong container — thay cron hệ thống, đọc lịch từ config/scheduler.yaml.

Bật khi docker-compose service `scheduler` chạy: `python -m scripts.scheduled_jobs`.
Mỗi job = 1 subprocess (cách ly môi trường, log riêng, mã thoát rõ ràng).
Job thất bại KHÔNG làm chết scheduler — lần chạy sau vẫn hoạt động.

Chạy từ gốc repo (thử trước khi lên container):
    python -m scripts.scheduled_jobs --run-once daily_pipeline   # chạy thử 1 job rồi thoát
    python -m scripts.scheduled_jobs                             # chạy scheduler thật
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys

# --- Windows an toàn: khi stdout/stderr là pipe/redirect, Python dùng codepage
# locale (vd cp1252) và sẽ crash khi in dấu tiếng Việt -> ép UTF-8 bắt buộc.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import yaml

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    APS_OK = True
except ImportError:
    APS_OK = False

CONFIG = Path("config/scheduler.yaml")
logger = logging.getLogger("fse.scheduler")


def load_jobs(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Thiếu {cfg_path} — đang chạy từ gốc repo chưa?")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not cfg.get("jobs"):
        raise ValueError(f"{cfg_path} không có job nào.")
    return cfg


def run_command(cmd: str) -> int:
    """Chạy 1 lệnh job bằng subprocess, log mã thoát."""
    logger.info("JOB START: %s", cmd)
    proc = subprocess.run(shlex.split(cmd))
    (logger.info if proc.returncode == 0 else logger.error)(
        "JOB END (%d): %s", proc.returncode, cmd)
    return proc.returncode


def build_trigger(name: str, job: dict) -> "CronTrigger":
    """Chuyển schedule 'HH:MM' (+ day_of_week/day tùy chọn) thành CronTrigger."""
    hour, minute = str(job["schedule"]).split(":")
    kwargs = {"hour": int(hour), "minute": int(minute)}
    if job.get("day_of_week"):
        kwargs["day_of_week"] = job["day_of_week"]
    if job.get("day"):
        kwargs["day"] = job["day"]
    logger.info("Đăng ký job '%s': %s %s", name, job["schedule"], job["command"])
    return CronTrigger(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scheduler định kỳ cho FSE")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--run-once", metavar="JOB",
                        help="Chạy thử một job theo tên rồi thoát (test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | fse.scheduler | %(message)s")

    cfg = load_jobs(Path(args.config))
    tz = cfg.get("timezone", "Asia/Ho_Chi_Minh")

    if args.run_once:
        job = cfg["jobs"].get(args.run_once)
        if not job:
            print(f"[LỖI] Không có job '{args.run_once}'. Có: {list(cfg['jobs'])}", file=sys.stderr)
            return 2
        return run_command(job["command"])

    if not APS_OK:
        print("[LỖI] Cài apscheduler: pip install apscheduler (xem requirements-ops.txt)",
              file=sys.stderr)
        return 2

    sched = BlockingScheduler(timezone=tz)
    for name, job in cfg["jobs"].items():
        if not job.get("enabled", False):
            logger.info("Job '%s' đang tắt (enabled: false) — bỏ qua.", name)
            continue
        sched.add_job(run_command, build_trigger(name, job),
                      args=[job["command"]], id=name,
                      misfire_grace_time=3600, coalesce=True)
    logger.info("Scheduler khởi động (tz=%s) — Ctrl+C để dừng.", tz)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler dừng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
