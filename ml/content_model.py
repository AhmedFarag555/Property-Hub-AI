import os
import sys
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

# ── path fix ──
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)


class ContentBasedRecommender:

    def __init__(self, model_path=None):
        # Always resolve to the ml/ directory — ignore caller's path if wrong
        resolved = model_path if model_path and os.path.exists(model_path) \
                else os.path.join(_ML_DIR, "model.pkl")

        self.content_model = joblib.load(resolved)

        self.df       = self.content_model["df"]
        self.nn       = self.content_model["nn"]
        self.fb       = self.content_model["feature_builder"]
        self.features = self.content_model["features"]

        self.id_to_index = {
            pid: idx for idx, pid in enumerate(self.df["property_id"])
        }

    # ----------------------------
    # GEO DISTANCE
    # ----------------------------
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    # ----------------------------
    # RECOMMEND BY PROPERTY ID
    # ----------------------------
    def recommend(self, property_id, top_n=5, user_profile=None):
        property_id = int(property_id)
        if property_id not in self.id_to_index:
            return []

        idx      = self.id_to_index[property_id]
        base_vec = self.features[idx].reshape(1, -1)
        distances, indices = self.nn.kneighbors(base_vec, n_neighbors=20)

        base      = self.df.iloc[idx]
        base_price = max(base["price"], 1)
        base_lat   = base["latitude"]
        base_lon   = base["longitude"]

        if pd.isna(base_lat) or pd.isna(base_lon):
            return []

        results, seen = [], set()
        for i, dist in zip(indices[0], distances[0]):
            prop = self.df.iloc[i]
            pid  = prop["property_id"]
            if pid == property_id or pid in seen:
                continue
            seen.add(pid)

            sim         = float(np.clip(1 / (1 + dist), 0, 1))
            price       = max(prop["price"], 1)
            price_score = float(np.clip(np.exp(-abs(price - base_price) / base_price), 0, 1))

            if pd.isna(prop.get("latitude")) or pd.isna(prop.get("longitude")):
                geo_score = 0.0
            else:
                geo_score = float(np.clip(
                    np.exp(-self.haversine_distance(base_lat, base_lon,
                                                    prop["latitude"], prop["longitude"]) / 15),
                    0, 1
                ))

            content_score = 0.7 * sim + 0.2 * geo_score + 0.1 * price_score

            if user_profile:
                if prop.get("city")          == user_profile.get("fav_city"):  content_score += 0.1
                if prop.get("property_type") == user_profile.get("fav_type"):  content_score += 0.1
                avg_p = user_profile.get("avg_price")
                if avg_p:
                    content_score += float(np.exp(-abs(prop["price"] - avg_p) / avg_p)) * 0.1

            content_score = min(content_score, 1.0)

            exclude_cols = ["image_urls", "updated_at"]
            prop_dict    = prop.drop(labels=[c for c in exclude_cols if c in prop.index]).to_dict()

            results.append({"property": prop_dict, "content_score": float(content_score)})

        results.sort(key=lambda x: x["content_score"], reverse=True)
        return results[:top_n]

    # ----------------------------
    # RECOMMEND FROM FEATURES
    # ----------------------------
    def recommend_from_features(
        self, city, area, compound, property_type, offering_type,
        area_sqm, bedrooms, bathrooms, amenities=None, top_n=5
    ):
        query_df = pd.DataFrame([{
            "price":          0,
            "area_sqm":       area_sqm,
            "bedrooms":       bedrooms,
            "bathrooms":      bathrooms,
            "price_per_sqm":  0,
            "amenities_count": len(amenities or []),
            "property_type":  property_type,
            "city":           city,
            "area":           area,
            "offering_type":  offering_type,
            "title":          "",
            "description":    "",
            "amenities":      amenities or []
        }])

        query_vec          = self.fb.transform(query_df)
        distances, indices = self.nn.kneighbors(query_vec, n_neighbors=min(20, len(self.df)))

        results, seen  = [], set()
        base_price     = self.df["price"].median()

        for dist, i in zip(distances[0], indices[0]):
            prop = self.df.iloc[i]
            pid  = prop["property_id"]
            if pid in seen:
                continue
            seen.add(pid)

            sim         = float(np.clip(1 / (1 + dist), 0, 1))
            price       = max(prop["price"], 1)
            price_score = float(np.clip(np.exp(-abs(price - base_price) / base_price), 0, 1))
            content_score = 0.8 * sim + 0.2 * price_score

            results.append({"property": prop.to_dict(), "content_score": float(content_score)})

        results.sort(key=lambda x: x["content_score"], reverse=True)
        return results[:top_n]


def run_test():
    model = ContentBasedRecommender("model.pkl")

    property_id = 71134541
    results = model.recommend(property_id, top_n=5)

    for i, r in enumerate(results, 1):
        print(f"\n🔹 Recommendation {i}")
        print("-" * 40)

        for key, value in r.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    run_test()