import asyncio
import logging
from playwright.async_api import async_playwright

try:
    from playwright_stealth import stealth_async
except ImportError:
    stealth_async = None

logger = logging.getLogger("AsyncFetcher")

class AsyncFetcher:
    def __init__(self, rate_limit_sec: float = 2.0, max_retries: int = 3):
        self.rate_limit_sec = rate_limit_sec
        self.max_retries = max_retries
        self.last_request_time = 0.0

    async def _enforce_rate_limit(self):
        """Tuân thủ giới hạn 1 request/2s/domain theo đúng Spec Section II.2."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_sec:
            await asyncio.sleep(self.rate_limit_sec - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()

    async def fetch_html(self, url: str) -> str:
        """Fetch nội dung trang web sử dụng Playwright Headless + Stealth."""
        for attempt in range(1, self.max_retries + 1):
            await self._enforce_rate_limit()
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        viewport={"width": 1920, "height": 1080}
                    )
                    page = await context.new_page()
                    
                    if stealth_async:
                        await stealth_async(page)
                    
                    # Chặn hình ảnh, font, css rác để tối ưu hiệu năng
                    await page.route(
                        "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,eot,ttf,otf}",
                        lambda route: route.abort()
                    )

                    # Tải trang theo DOMContentLoaded để vừa nhanh vừa giữ nguyên cấu trúc bài viết
                    await page.goto(url, wait_until='domcontentloaded', timeout=20000)
                    await asyncio.sleep(0.5)
                    
                    html_content = await page.content()
                    await browser.close()
                    return html_content
                    
            except Exception as e:
                logger.warning(f"Fetch thất bại ({url}) - Lần thử {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    logger.error(f"Đã vượt quá {self.max_retries} lần thử cho {url}.")
                    return ""
                await asyncio.sleep(2 ** attempt)
        return ""