import hashlib
import json
from datetime import datetime
from pathlib import Path


class Deduplicator:
    """Persistent dedup — lưu state ra JSON, giữ qua các lần chạy."""

    def __init__(self, state_path: str = "data/crawler_state.json",
                 max_size: int = 10000):
        self.state_path = Path(state_path)
        self.max_size = max_size
        self.seen_urls = set()
        self.seen_contents = set()
        self._load_state()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_state(self):
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.seen_urls = set(data.get("seen_urls", []))
                self.seen_contents = set(data.get("seen_contents", []))
            except Exception as e:
                print(f"  ⚠️  Load dedup state failed: {e}")

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        urls = list(self.seen_urls)[-self.max_size:]
        contents = list(self.seen_contents)[-self.max_size:]
        self.state_path.write_text(
            json.dumps(
                {"seen_urls": urls, "seen_contents": contents},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def is_duplicate_url(self, url: str) -> bool:
        url_hash = self._hash(url.strip().lower())
        if url_hash in self.seen_urls:
            return True
        self.seen_urls.add(url_hash)
        return False

    def is_duplicate_content(self, title: str, publish_time: str) -> bool:
        if not title or not publish_time:
            return False
        try:
            dt = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = publish_time[:10]

        key = f"{title.strip().lower()}_{date_str}"
        h = self._hash(key)
        if h in self.seen_contents:
            return True
        self.seen_contents.add(h)
        return False

    def save(self):
        """Gọi cuối pipeline để flush state."""
        self._save_state()