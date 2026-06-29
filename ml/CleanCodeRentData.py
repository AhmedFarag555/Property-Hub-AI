import pandas as pd
import re
import logging
from datetime import datetime
from sqlalchemy import create_engine

# =========================
# LOGGING
# =========================
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# LOAD DATA
# =========================
def load_data(path, encoding="utf-8-sig"):
    try:
        df = pd.read_csv(path, encoding=encoding)
        logging.info(f"Loaded data: {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    try:
        df = df.drop_duplicates(subset=["property_id"])
        df = df.dropna(subset=["property_id","all_image_url"])
        df = df[df['all_image_url'] != '[]']
        num_cols = ["bedrooms", "bathrooms", "price", "area_sqm", "latitude", "longitude"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # Bedrooms
        df = df[(df["bedrooms"] > 0) & (df["bedrooms"] <= 13)]

        # Bathrooms
        df = df[(df["bathrooms"] > 0) & (df["bathrooms"] <= 10)]
        
        df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
        df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

        df["bedrooms"] = df["bedrooms"].round().astype(int)
        df["bathrooms"] = df["bathrooms"].round().astype(int)
        df["offering_type"] = "rent-month"
        # TITLE
        df["title"] = (
            df["title"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # DESCRIPTION
        df["description"] = (
            df["description"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["description"] = df["description"].str.split().str[:100].str.join(" ")

        # property_type
        core_types = [
            "apartment",
            "villa",
            "chalet",
            "townhouse",
            "cabin",
            "duplex",
            "palace",
            "land",
            "penthouse"
        ]
        
        mapping = {
            "ivilla": "villa",
            "hotel apartment": "apartment",
            "roof": "apartment",
            "twin house": "townhouse"
        }
        
        df["property_type"] = (
            df["property_type"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace(mapping)
        )
        
        df["property_type"] = df["property_type"].where(
            df["property_type"].isin(core_types),
            "other"
        )
        
        df["property_type"] = pd.Categorical(
            df["property_type"],
            categories=core_types + ["other"],
            ordered=True
        )
        # COORDINATES

        df = df[
            df["latitude"].notna() &
            df["longitude"].notna() &
            (df["latitude"] != 0) &
            (df["longitude"] != 0)
        ]
        
        df = df[
            (df["latitude"].between(22, 32)) &   # مصر
            (df["longitude"].between(25, 36))
        ]
        
        
        # last_refreshed
        df["last_refreshed_at"] = pd.to_datetime(df["last_refreshed_at"], errors="coerce")
        df = df.rename(columns={"last_refreshed_at": "updated_at"})
        df["updated_at"] = df["updated_at"].fillna(pd.Timestamp.now())

        # AREA & PRICE
        
        df = df[df["price"] < 900000]
        df = df.dropna(subset=[
            "price", "area_sqm", "location_full"
        ])
        
        df = df[(df["price"] > 0) & (df["area_sqm"] > 0)]
        df = df[(df["area_sqm"] >= 50) & (df["area_sqm"] <= 5000)]
        
        df["price_per_sqm"] = df["price"] / df["area_sqm"]
        
        df["price_per_sqm"] = df["price_per_sqm"].replace([float("inf")], None)
        df["price_per_sqm_display"] = df["price_per_sqm"].apply(
            lambda x: f"{int(x):,} EGP/m²" if pd.notnull(x) else None
        )
        

        logging.info(f"After cleaning: {len(df)} rows")
        return df

    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise



# =========================
# FEATURE ENGINEERING
# =========================
def feature_engineering(df):

    # LOCATION
    
    loc_split = df["location_full"].str.split(",", expand=True)

    df["compound"] = loc_split[0].str.strip()
    
    df["area"] = loc_split.apply(
        lambda row: row.dropna().iloc[-2] if len(row.dropna()) > 1 else None,
        axis=1
    )
    df["area"] = df["area"].str.lower().str.strip()


    df["compound"] = (
        df["compound"]
        .str.lower()
        .str.replace(r"\b(city|icity)\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    
    # 🟢 AREA = آخر جزء بعد آخر فاصلة
    df["city"] = loc_split.apply(
        lambda row: row.dropna().iloc[-1] if len(row.dropna()) > 0 else None,
        axis=1
    )

    # 🧹 تنظيف الـ area
    df["city"] = (
        df["city"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    # AMENITIES
    df["amenities"] = (
        df["amenities"]
        .fillna("")
        .str.lower()
        .str.split(",")
        .apply(lambda x: [i.strip() for i in x if i.strip()])
    )
    

    # DROP USELESS
    df = df.drop(columns=[
        "broker_name",
        "broker_email",
        "broker_phone",
        "currency",
        "listed_date",
        "listing_id",
        "image_urls"
    ], errors="ignore")
    
    df = df.reset_index(drop=True)
    
    return df

# =========================
# SAVE TO DATA
# =========================
def connect_db():
    engine = create_engine(
        "mysql+mysqlconnector://root:1234@localhost/real_estate_db"
    )
    return engine

def save_to_mysql(df):
    engine = connect_db()

    df.to_sql(
        name="properties_rent",
        con=engine,
        if_exists="replace",   # يضيف كل مرة
        index=False
    )

    print("Data saved to MySQL ✅")
    
    


def run_pipeline():
    logging.info("Pipeline started")

    df = load_data("ml\RentData.csv", encoding="utf-8-sig")
    df = clean_data(df)
    df = feature_engineering(df)
    #df.to_csv("rentdataclened.csv", index=False, encoding="utf-8-sig")
    # تحويل list → string قبل التخزين
    df["amenities"] = df["amenities"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )

    save_to_mysql(df) 
    logging.info(f"Final dataset size: {df.shape}")
    logging.info("Pipeline finished")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_pipeline()