import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

class DiscoveryEngine:
    def __init__(self, sources: list):
        self.sources = sources
        # Ánh xạ URL sitemap/RSS hoặc cấu trúc Category của 04 nguồn
        self.configs = {
            "cafef": {"url": "https://cafef.vn/tai-chinh-ngan-hang.rss", "type": "rss"},
            "vietstock": {"url": "https://vietstock.vn/tai-chinh.htm", "type": "html_category"},
            "vietnamfinance": {"url": "https://vietnamfinance.vn/tai-chinh-ngan-hang.htm", "type": "html_category"},
            "thoibaotaichinhvietnam": {"url": "https://thoibaotaichinhvietnam.vn/tai-chinh", "type": "html_category"}
        }

    async def discover_urls(self) -> list:
        discovered = []
        for source in self.sources:
            if source not in self.configs:
                continue
                
            config = self.configs[source]
            try:
                response = requests.get(config["url"], timeout=10)
                if config["type"] == "rss":
                    soup = BeautifulSoup(response.content, features="xml")
                    items = soup.find_all("item")
                    for item in items:
                        link = item.find("link").text if item.find("link") else ""
                        if link:
                            discovered.append(self._build_item(link, source, "RSS"))
                else:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # CSS selector giả định trích xuất tất cả thẻ <a> có href
                    for a_tag in soup.find_all('a', href=True):
                        link = a_tag['href']
                        # Chỉ lấy các bài viết có định dạng bài báo
                        if ("htm" in link or "html" in link) and len(link) > 20:
                            if link.startswith("/"):
                                domain_base = config["url"].split("/")[2]
                                link = f"https://{domain_base}{link}"
                            discovered.append(self._build_item(link, source, "Category"))
            except Exception as e:
                print(f"Lỗi Discovery nguồn {source}: {e}")
                
        return discovered

    def _build_item(self, url: str, source: str, method: str) -> dict:
        return {
            "url": url,
            "source": source,
            "category": method,
            "discovered_time": datetime.now().isoformat()
        }