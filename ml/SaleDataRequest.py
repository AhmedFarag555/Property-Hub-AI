#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import os

headers = {
    "User-Agent": "Mozilla/5.0"
}

BASE_URL = "https://www.propertyfinder.eg/en/buy/properties-for-sale.html"
MAX_PAGES = 50

FILE_PATH = "ml\SaleData.csv"


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


def scrape_data():
    all_rows = []

    for page in range(1, MAX_PAGES + 1):
        url = f"{BASE_URL}?page={page}"
        print(f"Scraping page {page}")

        try:
            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            rows = extract_from_next_data(soup)

            if not rows:
                break

            all_rows.extend(rows)
            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            continue

    df = pd.DataFrame(all_rows)

    

    return df


def merge_and_save(df_new):
    if os.path.exists(FILE_PATH):
        df_old = pd.read_csv(FILE_PATH)

        # الجديد فوق 🔥
        df_all = pd.concat([df_new, df_old], ignore_index=True)
    else:
        df_all = df_new

    # إزالة التكرار (الجديد يكسب)
    df_all["property_id"] = df_all["property_id"].astype(str)
    df_all.drop_duplicates(subset=["property_id"], keep="first", inplace=True)
    #limit
    df_all = df_all.sort_values("listed_date", ascending=False).head(50000)


    df_all.to_csv(FILE_PATH, index=False)

    print("Saved ✅ Total rows:", df_all.shape[0])


if __name__ == "__main__":
    print("Starting scraping...")

    df_new = scrape_data()

    print("New rows:", df_new.shape)

    merge_and_save(df_new)


# In[ ]:




