"""APScheduler wrapper cho pipeline — chạy tự động mỗi N phút."""
import asyncio
from apscheduler.schedulers.blocking import BlockingScheduler

from .pipeline import run_pipeline


DEFAULT_SOURCES = ["cafef", "vnexpress"]
DEFAULT_INTERVAL_MINUTES = 30


def _job(sources: list):
    """Job wrapper: chạy async pipeline trong sync context."""
    try:
        asyncio.run(run_pipeline(sources))
    except Exception as e:
        print(f"❌ Job failed: {e}")


def run_forever(sources: list = None,
                interval_minutes: int = DEFAULT_INTERVAL_MINUTES):
    sources = sources or DEFAULT_SOURCES
    print(f"\n🚀 Scheduler started")
    print(f"   sources = {sources}")
    print(f"   interval = {interval_minutes} min")
    print(f"   Ctrl+C để dừng\n")

    # Chạy ngay 1 lần đầu
    _job(sources)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _job,
        trigger="interval",
        minutes=interval_minutes,
        args=[sources],
        id="fse_crawler",
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n\n⛔ Scheduler stopped")


if __name__ == "__main__":
    run_forever()