import sys
import subprocess
import requests
from data_cache import cache

python = sys.executable

# ✅ السيرفر الشغال (uvicorn) — غيّر البورت لو مختلف
API_BASE = "http://127.0.0.1:8000"
CACHE_REFRESH_SECRET = "propertyhub-internal"  # نفس القيمة في .env / CACHE_REFRESH_SECRET


def run(script):
    print(f"\n▶ Running {script}...")

    result = subprocess.run([python, script])

    if result.returncode != 0:
        print(f"❌ Failed: {script}")
        sys.exit(1)

    print(f"✅ Finished: {script}")


# ---------------- PIPELINE ----------------
print("🚀 Starting full ETL + ML pipeline...")

# 1. SCRAPING
#run("ml\\SaleDataRequest.py")
#run("ml\\RentDataRequest.py")
#run("ml\\image_sale.py")
#run("ml\\image_rent.py")

# 2. CLEANING
run("ml\\CleanCodeSaleData.py")
run("ml\\CleanCodeRentData.py")

# 3. MERGING
run("ml\\merge_data.py")
run("ml\\clean_data.py")

# 4. TRAINING
run("ml/trainer.py")
run("ml/ranker_trainer.py")

# 5. REFRESH CACHE (LOCAL — لو الـ pipeline نفسه بيستخدم cache مباشرة)
print("\n🔄 Refreshing local cache instance...")
cache.refresh()

# 6. ✅ REFRESH CACHE على السيرفر الشغال (uvicorn) — ده اللي الموقع بيستخدمه فعلياً
print("\n🌐 Refreshing live server cache...")
try:
    resp = requests.post(
        f"{API_BASE}/properties/admin/refresh-cache",
        params={"secret": CACHE_REFRESH_SECRET},
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ Server cache refreshed — {data.get('properties_count', '?')} properties now live")
    else:
        print(f"⚠️  Server refresh returned {resp.status_code}: {resp.text}")
        print("   (السيرفر شغال؟ تأكد إن uvicorn يعمل على نفس البورت)")
except requests.exceptions.RequestException as e:
    print(f"⚠️  Could not reach live server: {e}")
    print("   الداتا اتحدثت في الـ DB لكن السيرفر هياخدها بس عند أول refresh/restart")

print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")