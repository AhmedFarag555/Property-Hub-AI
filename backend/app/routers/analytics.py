from fastapi import APIRouter
import pandas as pd
import sys, os, importlib, importlib.util

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
ML_DIR   = os.path.join(BASE_DIR, "ml")
for p in [BASE_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ✅ نفس الـ singleton بالظبط — مش instance جديد
def get_cache():
    if "data_cache" in sys.modules:
        return sys.modules["data_cache"].cache
    elif "ml.data_cache" in sys.modules:
        return sys.modules["ml.data_cache"].cache
    else:
        spec = importlib.util.spec_from_file_location(
            "data_cache", os.path.join(ML_DIR, "data_cache.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["data_cache"] = mod
        spec.loader.exec_module(mod)
        return mod.cache

router = APIRouter(prefix="/analytics", tags=["Analytics"])


def get_df():
    return get_cache().properties


# ── GET /analytics/ ───────────────────────────────────────────────────────────
@router.get("/")
def get_analytics():
    df = get_df()

    total_properties = len(df)

    rent_count = len(
        df[df["offering_type"].astype(str).str.lower().str.contains("rent", na=False)]
    )

    avg_price = float(df["price"].mean())
    max_price = float(df["price"].max())
    min_price = float(df["price"].min())

    cities_count = df["city"].nunique()

    city_counts = df["city"].value_counts()
    top_city = {
        "name":  city_counts.index[0] if len(city_counts) else None,
        "count": int(city_counts.iloc[0]) if len(city_counts) else 0
    }

    property_type_counts = df["property_type"].value_counts()
    top_property_type = {
        "name":  property_type_counts.index[0] if len(property_type_counts) else None,
        "count": int(property_type_counts.iloc[0]) if len(property_type_counts) else 0
    }

    city_distribution = [
        {"city": city, "count": int(count)}
        for city, count in city_counts.items()
    ]

    property_type_distribution = [
        {"type": prop_type, "count": int(count)}
        for prop_type, count in property_type_counts.items()
    ]

    price_bins   = [0, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, float("inf")]
    price_labels = ["0-500K", "500K-1M", "1M-2M", "2M-5M", "5M-10M", "10M+"]

    price_ranges = pd.cut(df["price"], bins=price_bins, labels=price_labels, include_lowest=True)
    price_distribution = [
        {"label": str(label), "count": int(count)}
        for label, count in price_ranges.value_counts().sort_index().items()
    ]

    bedrooms_distribution = [
        {"bedrooms": int(bedroom), "count": int(count)}
        for bedroom, count in df["bedrooms"].value_counts().sort_index().items()
    ]

    return {
        "total_properties":          total_properties,
        "rent_count":                rent_count,
        "avg_price":                 avg_price,
        "max_price":                 max_price,
        "min_price":                 min_price,
        "cities_count":              cities_count,
        "top_city":                  top_city,
        "top_property_type":         top_property_type,
        "city_distribution":         city_distribution,
        "property_type_distribution":property_type_distribution,
        "price_distribution":        price_distribution,
        "bedrooms_distribution":     bedrooms_distribution
    }


# ── GET /analytics/dashboard ──────────────────────────────────────────────────
@router.get("/dashboard")
def dashboard():
    cache        = get_cache()
    properties_df = cache.properties
    users_df      = cache.users

    total_properties = len(properties_df)
    avg_price        = float(properties_df["price"].mean()) if "price" in properties_df.columns else 0
    cities_count     = properties_df["city"].nunique()      if "city"  in properties_df.columns else 0
    registered_users = len(users_df)

    latest_properties = properties_df.head(5)
    recent_activity = [
        {
            "text": f"New property listed in {row['city']}" if "city" in properties_df.columns else "New property listed",
            "time": "Recently"
        }
        for _, row in latest_properties.iterrows()
    ]

    return {
        "total_properties": total_properties,
        "avg_price":        avg_price,
        "cities_count":     cities_count,
        "registered_users": registered_users,
        "recent_activity":  recent_activity
    }