"""CLI entry point cho crawler.

Usage:
    python -m scripts.run_collector                      # chạy 1 lần, tất cả sources
    python -m scripts.run_collector --sources cafef      # chỉ cafef
    python -m scripts.run_collector --daemon             # tự động mỗi 30 phút
    python -m scripts.run_collector --daemon --interval 5
"""
import argparse
import asyncio

from src.crawler.pipeline import run_pipeline
from src.crawler.scheduler import run_forever, DEFAULT_SOURCES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help="Nguồn cách nhau bởi dấu phẩy",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Chạy nền theo interval",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Interval phút (chỉ dùng với --daemon)",
    )
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")]

    if args.daemon:
        run_forever(sources, args.interval)
    else:
        asyncio.run(run_pipeline(sources))


if __name__ == "__main__":
    main()