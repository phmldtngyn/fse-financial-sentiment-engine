import sys
import asyncio

def check_imports():
    print("=== 1. KIỂM TRA THƯ VIỆN PYTHON ===")
    modules = {
        "trafilatura": "trafilatura",
        "playwright": "playwright",
        "playwright_stealth": "playwright_stealth",
        "bs4": "beautifulsoup4",
        "requests": "requests"
    }
    
    all_passed = True
    for mod_name, pkg_name in modules.items():
        try:
            __import__(mod_name)
            print(f"  [OK] {pkg_name} đã sẵn sàng.")
        except ImportError:
            print(f"  [X] {pkg_name} CHƯA ĐƯỢC CÀI ĐẶT!")
            all_passed = False
            
    return all_passed

async def test_playwright_and_trafilatura():
    print("\n=== 2. THỬ NGHIỆM PLAYWRIGHT & TRAFILATURA ===")
    from playwright.async_api import async_playwright
    import trafilatura

    test_url = "https://example.com"
    print(f"Đang thử fetch URL: {test_url}...")

    try:
        async with async_playwright() as p:
            # 1. Khởi chạy Chromium Headless
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # 2. Điều hướng trang
            await page.goto(test_url, wait_until="domcontentloaded", timeout=10000)
            html_content = await page.content()
            await browser.close()
            
            print("  [OK] Playwright Chromium hoạt động bình thường!")

            # 3. Thử trích xuất bằng Trafilatura
            text = trafilatura.extract(html_content)
            if text:
                print("  [OK] Trafilatura trích xuất text thành công!")
                print(f"  Nội dung mẫu: '{text.strip()[:60]}...'")
            else:
                print("  [X] Trafilatura không trích xuất được nội dung.")

    except Exception as e:
        print(f"  [X] Lỗi trong quá trình thử nghiệm Playwright: {e}")
        print("      Lưu ý: Nếu báo lỗi Chromium missing, hãy chạy: playwright install chromium")

if __name__ == "__main__":
    if check_imports():
        asyncio.run(test_playwright_and_trafilatura())
    else:
        print("\nVui lòng cài đặt các thư viện còn thiếu trước khi tiếp tục!")