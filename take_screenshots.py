import asyncio
import os
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # set viewport
        await page.set_viewport_size({"width": 1400, "height": 1080})

        print("Navigating to http://localhost:5000 ...")
        await page.goto("http://localhost:5000")

        print("Waiting for page load...")
        upload_input = page.locator('input[type="file"]')
        if await upload_input.count() > 0:
            image_path = os.path.abspath(r"c:\antigravity_projects\svd compressor\kodak_images\kodim01.png")
            print(f"Uploading image: {image_path}")
            await upload_input.set_input_files(image_path)
            
            compress_btn = page.locator('#btn-compress')
            await compress_btn.click()
            
            print("Waiting for charts section to become visible...")
            await page.wait_for_selector('#charts-section:not(.hidden)', timeout=30000)
            
            # Wait for charts to animate
            await page.wait_for_timeout(3000)
            
            screenshots_dir = r"c:\antigravity_projects\svd compressor\screenshots"
            os.makedirs(screenshots_dir, exist_ok=True)
            
            # Original Image 
            await page.locator('#img-original').screenshot(path=os.path.join(screenshots_dir, 'fig1_original.png'))
            
            await page.locator('#chart-sv').locator('xpath=ancestor::div[contains(@class, "chart-card")]').first.screenshot(path=os.path.join(screenshots_dir, 'fig2_svd_basic_ranks.png'))
            
            await page.locator('#chart-energy').locator('xpath=ancestor::div[contains(@class, "chart-card")]').first.screenshot(path=os.path.join(screenshots_dir, 'fig3_svd_adaptive_energy.png'))
            
            await page.locator('#chart-psnr').locator('xpath=ancestor::div[contains(@class, "chart-card")]').first.screenshot(path=os.path.join(screenshots_dir, 'fig5_metrics_vs_k.png'))

            await page.locator('#img-error-map').locator('xpath=ancestor::div[contains(@class, "analysis-item")]').first.screenshot(path=os.path.join(screenshots_dir, 'fig6_error_maps.png'))
            
            await page.locator('#img-heatmap').locator('xpath=ancestor::div[contains(@class, "analysis-item")]').first.screenshot(path=os.path.join(screenshots_dir, 'fig8_adaptive_heatmap.png'))
            
            # Benchmark chart
            bench_btn = page.locator('#btn-benchmark')
            if await bench_btn.count() > 0:
                await bench_btn.click()
                await page.wait_for_selector('#card-benchmark:not([style*="display: none"])', timeout=30000)
                await page.wait_for_timeout(3000)
                await page.locator('#chart-benchmark').locator('xpath=ancestor::div[contains(@class, "chart-card")]').first.screenshot(path=os.path.join(screenshots_dir, 'fig7_benchmark_timing.png'))

            print("Screenshots taken.")
        else:
            print("Could not find file input.")
            
        await browser.close()

asyncio.run(main())
