import requests
from bs4 import BeautifulSoup
from datetime import datetime


class DiscoveryEngine:
    def __init__(self, sources: list):
        self.sources = sources
        # Mỗi source có thể có nhiều feeds (đa chuyên mục)
        # Format mới: list các dict với `category` semantic
        self.configs = {
            "cafef": [
                {"url": "https://cafef.vn/doanh-nghiep.rss",
                 "type": "rss", "category": "company"},
                {"url": "https://cafef.vn/vi-mo-dau-tu.rss",
                 "type": "rss", "category": "macro"},
                {"url": "https://cafef.vn/thi-truong-chung-khoan.rss",
                 "type": "rss", "category": "market"},
                {"url": "https://cafef.vn/tai-chinh-ngan-hang.rss",
                 "type": "rss", "category": "market"},
            ],
            "vnexpress": [
                {"url": "https://vnexpress.net/rss/kinh-doanh.rss",
                 "type": "rss", "category": "market"},
            ],
            # Giữ nguyên 3 nguồn cũ (HTML category) để không break
            "vietstock": [
                {"url": "https://vietstock.vn/tai-chinh.htm",
                 "type": "html_category", "category": "market"},
            ],
            "vietnamfinance": [
                {"url": "https://vietnamfinance.vn/tai-chinh-ngan-hang.htm",
                 "type": "html_category", "category": "market"},
            ],
            "thoibaotaichinhvietnam": [
                {"url": "https://thoibaotaichinhvietnam.vn/tai-chinh",
                 "type": "html_category", "category": "market"},
            ],
        }

        # User-Agent tránh bị block khi poll thường xuyên
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FSE-Research-Crawler/1.0)"
        }

    async def discover_urls(self) -> list:
        discovered = []
        for source in self.sources:
            if source not in self.configs:
                print(f"  ⚠️  Unknown source: {source}")
                continue

            feeds = self.configs[source]
            if isinstance(feeds, dict):
                feeds = [feeds]  # backward compat nếu có dict cũ

            for config in feeds:
                try:
                    response = requests.get(
                        config["url"], headers=self.headers, timeout=15
                    )
                    response.raise_for_status()
                    method = "rss" if config["type"] == "rss" else "html"
                    category = config.get("category", "unknown")

                    if config["type"] == "rss":
                        soup = BeautifulSoup(response.content, features="xml")
                        for item in soup.find_all("item"):
                            link_tag = item.find("link")
                            if link_tag and link_tag.text:
                                discovered.append(
                                    self._build_item(
                                        link_tag.text.strip(), source, "rss", category
                                    )
                                )
                    else:
                        soup = BeautifulSoup(response.content, "html.parser")
                        for a_tag in soup.find_all("a", href=True):
                            link = a_tag["href"]
                            if ("htm" in link or "html" in link) and len(link) > 20:
                                if link.startswith("/"):
                                    domain_base = config["url"].split("/")[2]
                                    link = f"https://{domain_base}{link}"
                                discovered.append(
                                    self._build_item(link, source, "html", category)
                                )
                except Exception as e:
                    print(f"  ⚠️  Discovery {source}/{config.get('category','?')}: {e}")

        return discovered

    def _build_item(self, url: str, source: str, method: str, category: str) -> dict:
        return {
            "url": url,
            "source": source,
            "category": category,          # semantic: company/macro/market
            "discovery_method": method,    # rss/html
            "discovered_time": datetime.now().isoformat(),
        }