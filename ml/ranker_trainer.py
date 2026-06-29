import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import numpy as np
from ranking_dataset_builder import RankingDatasetBuilder
from db import engine

class RankerTrainer:

    def train(self):

        print("Loading data...")

        interactions = pd.read_sql("SELECT * FROM interactions", engine)
        properties = pd.read_sql("SELECT * FROM properties_clean", engine)

        print("Building dataset...")

        builder = RankingDatasetBuilder()
        df = builder.build(interactions, properties)

        print("Dataset shape:", df.shape)

        # ❌ safety check
        if df.empty:
            print("❌ Empty dataset - stopping training")
            return

        # ---------------- LABEL ----------------
        y = df["label"].astype(int)

        # ---------------- FEATURES ----------------
        num_features = [

            "price",
            "area_sqm",
            "bedrooms",
            "bathrooms",

            "similarity_score",
            "price_affinity",
            "popularity",
            "recency_score",
            "city_match",
            "recent_city_match",
            "session_recency",
            "type_match"
        ]
        cat_features = ["city", "area", "property_type", "offering_type", "compound"]

        df = df.sort_values("user_id")

        # ---------------- NUMERIC ----------------
        for col in num_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[num_features] = df[num_features].fillna(0)

    
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(df[num_features])

        # ---------------- CATEGORICAL ----------------
        encoder = OneHotEncoder(handle_unknown="ignore")

        X_cat = encoder.fit_transform(
            df[cat_features].fillna("unknown")
        )

        # ---------------- FINAL MATRIX ----------------
        X = hstack([X_num, X_cat]).tocsr()

        # ---------------- GROUP FIX ----------------
        group = df.groupby("user_id").size().values

        print("Users:", len(group))

        # ---------------- MODEL ----------------
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31
        )

        print("Training model...")

        model.fit(
            X,
            y,
            group=group
        )

        print("Saving model...")

        joblib.dump(
            {
                "model": model,
                "encoder": encoder,
                "scaler": scaler,
                "num_features": num_features,
                "cat_features": cat_features
            },
            "ml/ranker.pkl"
        )

        print("✅ Ranker trained successfully")
        
        
if __name__ == "__main__":

    trainer = RankerTrainer()
    trainer.train()