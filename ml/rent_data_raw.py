#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time

headers = {
    "User-Agent": "Mozilla/5.0"
}

BASE_URLS = [
    "https://www.propertyfinder.eg/en/rent/properties-for-rent.html"
]

MAX_PAGES = 600

all_rows = []

def extract_from_next_data(soup):
    script = soup.find("script", {"id": "__NEXT_DATA__"})

    if not script:
        return []

    data = json.loads(script.string)

    listings = (
        data.get("props", {})
        .get("pageProps", {})
        .get("searchResult", {})
        .get("listings", [])
    )

    rows = []

    for item in listings:
        prop = item.get("property") or {}

        location = prop.get("location") or {}
        coordinates = location.get("coordinates") or {}
        broker = prop.get("broker") or {}
        price = prop.get("price") or {}
        price_area = prop.get("price_per_area") or {}
        size = prop.get("size") or {}

        images = prop.get("images") or []

        # تعديل: أخذ الصورة المتوسطة أو الصغيرة لكل صورة
        image_urls = []
        for img in images:
            if isinstance(img, dict):
                link = img.get("medium") or img.get("small")
                if link:
                    image_urls.append(link)

        rows.append({
            "property_id": prop.get("id"),
            "listing_id": prop.get("listing_id"),
            "title": prop.get("title"),
            "description": prop.get("description"),
            "property_type": prop.get("property_type"),
            "offering_type": prop.get("offering_type"),
            "price": price.get("value"),
            "currency": price.get("currency"),
            "price_period": price.get("period"),
            "price_per_sqm": price_area.get("price"),
            "bedrooms": prop.get("bedrooms_value"),
            "bathrooms": prop.get("bathrooms_value"),
            "area_sqm": size.get("value"),
            "location_full": location.get("full_name"),
            "latitude": coordinates.get("lat"),
            "longitude": coordinates.get("lon"),
            "amenities": ", ".join(prop.get("amenity_names", [])),
            "broker_name": broker.get("name"),
            "broker_email": broker.get("email"),
            "broker_phone": broker.get("phone"),
            "listed_date": prop.get("listed_date"),
            "last_refreshed_at": prop.get("last_refreshed_at"),
            "image_urls": ", ".join(image_urls),
            "share_url": prop.get("share_url")
        })

    return rows


for base_url in BASE_URLS:
    for page in range(1, MAX_PAGES + 1):
        url = f"{base_url}?page={page}"

        print(f"\nScraping page {page}: {url}")

        response = requests.get(url, headers=headers, timeout=60)

        print("Status:", response.status_code)

        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        rows = extract_from_next_data(soup)

        print("Found listings:", len(rows))

        all_rows.extend(rows)

        time.sleep(2)


df = pd.DataFrame(all_rows)

print("\nFinal shape:", df.shape)
print(df.head())

# عرض رابط أول صورة لأول عقار (لو موجودة)
if not df.empty and df.loc[0, "image_urls"]:
    first_image_link = df.loc[0, "image_urls"].split(",")[0]
    print("\nFirst image URL of the first property:", first_image_link)

df.to_csv("propertyfinder_real_full_rent_features_with_images.csv", index=False)

print("\nCSV saved successfully")


# In[ ]:




