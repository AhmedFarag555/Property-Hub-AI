from playwright.sync_api import sync_playwright
import os
import re
import time
import pandas as pd
import json

# ======================
# 📥 LOAD CSV
# ======================
df = pd.read_csv("ml\RentData.csv")

if "all_image_url" not in df.columns:
    df["all_image_url"] = None
    
    
df = df.drop_duplicates(subset='property_id', keep='first')

# ✅ شغل بس على الفاضي
rows = df[
    (df["all_image_url"].isna()) &
    (df["share_url"].notna()) &
    (df["share_url"].astype(str).str.startswith("http"))
][["property_id", "share_url"]].values.tolist()

print("Rows to process:", len(rows))


# ======================
# 🧠 HELPER
# ======================
def get_resolution(url):
    match = re.search(r'(\d{3,4})x(\d{3,4})', url)
    if match:
        return int(match.group(1)) * int(match.group(2))
    return 999999


# ======================
# 🚀 PLAYWRIGHT INIT (ONCE)
# ======================
p = sync_playwright().start()

browser = p.chromium.launch_persistent_context(
    user_data_dir="profile",
    headless=False,
    args=[
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-dev-shm-usage"
    ],
    viewport={"width": 1280, "height": 800}
)

page = browser.new_page()


# ======================
# 🧠 SCRAPER FUNCTION
# ======================
def scrape_images(url):

    images = {}

    def capture(response):
        link = response.url

        if "static.shared.propertyfinder.eg/media/images/listing" in link:
            if link.endswith((".jpg", ".webp", ".png")):

                base_id = link.split("/")[-2]

                if base_id not in images or get_resolution(link) > get_resolution(images[base_id]):
                    images[base_id] = link

    page.on("response", capture)

    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(1500)

    page.mouse.wheel(0, 1500)
    page.wait_for_timeout(1000)

    # open gallery
    try:
        page.locator("button[data-testid='gallery-image-button']").click(timeout=3000)
    except:
        try:
            page.locator("img").first.click(force=True)
        except:
            page.evaluate("document.querySelector('img')?.click()")

    page.wait_for_timeout(2000)

    # navigate images (light)
    for _ in range(5):
        page.keyboard.press("ArrowRight")
        time.sleep(0.3)

    page.remove_listener("response", capture)

    return list(images.values())


# ======================
# 🔁 LOOP
# ======================
counter = 0

for i, (property_id, url) in enumerate(rows):

    print(f"\n[{i+1}/{len(rows)}] Property ID: {property_id}")

    try:
        img_list = scrape_images(url)

        clean_list = [
            img for img in img_list
            if isinstance(img, str) and img.startswith("http")
        ]

        # ======================
        # 💾 UPDATE DF
        # ======================
        df.loc[df["property_id"] == property_id, "all_image_url"] = json.dumps(clean_list)

        print("Saved ✔ | Images:", len(clean_list))

        # ======================
        # 💾 AUTO SAVE (EVERY 5 ROWS)
        # ======================
        counter += 1

        if counter % 5 == 0:
            df.to_csv(
                "RentData.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print("💾 Batch Saved")

    except Exception as e:
        print("Error:", e)

# ======================
# 💾 FINAL SAVE
# ======================
df.to_csv(
    "RentData.csv",
    index=False,
    encoding="utf-8-sig"
)

browser.close()
p.stop()

print("\nDONE ✔ CSV Updated")