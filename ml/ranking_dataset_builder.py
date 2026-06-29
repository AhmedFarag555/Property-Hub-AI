import pandas as pd
import numpy as np


ACTION_LABELS = {
    "view": 1,
    "click": 2,
    "favorite": 6,
    "contact": 12
}


class RankingDatasetBuilder:

    def build(self, interactions, properties):

        df = interactions.merge(
            properties,
            on="property_id",
            how="left"
        )

        df["label"] = df["action"].map(ACTION_LABELS).fillna(0)

        # ====================================================
        # USER FEATURES
        # ====================================================

        user_stats = df.groupby("user_id").agg({
            "price": "mean",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()

        user_stats.columns = [
            "user_id",
            "user_avg_price",
            "user_avg_lat",
            "user_avg_lon"
        ]

        df = df.merge(user_stats, on="user_id", how="left")

        # ====================================================
        # PRICE AFFINITY
        # ====================================================

        df["price_affinity"] = np.exp(
            -abs(df["price"] - df["user_avg_price"]) /
            (df["user_avg_price"] + 1e-9)
        )

        # ====================================================
        # POPULARITY
        # ====================================================

        popularity = (
            interactions.groupby("property_id")
            .size()
            .reset_index(name="popularity")
        )

        df = df.merge(
            popularity,
            on="property_id",
            how="left"
        )

        # ====================================================
        # RECENCY
        # ====================================================

        if "created_at" in df.columns:

            days_old = (
                pd.Timestamp.now() -
                pd.to_datetime(df["created_at"])
            ).dt.days

            df["recency_score"] = np.exp(
                -days_old / 30
            )

        else:
            df["recency_score"] = 0.5

        # ====================================================
        # CITY / TYPE MATCH
        # ====================================================

        fav_city = (
            df.groupby("user_id")["city"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
            .reset_index(name="fav_city")
        )

        fav_type = (
            df.groupby("user_id")["property_type"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
            .reset_index(name="fav_type")
        )

        df = df.merge(fav_city, on="user_id", how="left")
        df = df.merge(fav_type, on="user_id", how="left")

        df["city_match"] = (
            df["city"] == df["fav_city"]
        ).astype(int)

        df["type_match"] = (
            df["property_type"] == df["fav_type"]
        ).astype(int)
        
        # ====================================================
        # RECENT CITY MATCH
        # ====================================================

        df["recent_city_match"] = df["city_match"]

        # ====================================================
        # SESSION RECENCY
        # ====================================================

        df["session_recency"] = 0

        # ====================================================
        # PLACEHOLDER SIMILARITY
        # ====================================================

        df["similarity_score"] = 0.5

        # ====================================================
        # FILL MISSING
        # ====================================================

        feature_cols = [

            "price",
            "area_sqm",
            "bedrooms",
            "bathrooms",

            "similarity_score",
            "price_affinity",
            "popularity",
            "recency_score",
            "city_match",
            "type_match",
            "recent_city_match",
            "session_recency",
            "city",
            "area",
            "property_type",
            "offering_type",
            "compound"
        ]

        for col in feature_cols:

            if col in df.columns:

                if df[col].dtype == "object":
                    df[col] = df[col].fillna("unknown")

                else:
                    df[col] = df[col].fillna(0)

        return df[[
            "user_id",
            "property_id",

            "price",
            "area_sqm",
            "bedrooms",
            "bathrooms",

            "similarity_score",
            "price_affinity",
            "popularity",
            "recency_score",
            "city_match",
            "type_match",
            "recent_city_match",
            "session_recency",
            "city",
            "area",
            "property_type",
            "offering_type",
            "compound",

            "label"
        ]]