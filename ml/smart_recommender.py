import os
import sys
import joblib
import pandas as pd
import numpy as np

from scipy.sparse import hstack

# ── path fix: allow running from backend OR standalone ──
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from user_vector_builder import UserVectorBuilder
from candidate_generator import CandidateGenerator
from data_cache import cache


class SmartRecommender:

    def __init__(self):

        BASE_DIR = _ML_DIR
        MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
        RANKER_PATH = os.path.join(BASE_DIR, "ranker.pkl")

        self.content_model = joblib.load(MODEL_PATH)
        self.ranker_data   = joblib.load(RANKER_PATH)

        self.scaler      = self.ranker_data["scaler"]
        self.ranker      = self.ranker_data["model"]
        self.encoder     = self.ranker_data["encoder"]
        self.num_features = self.ranker_data["num_features"]
        self.cat_features = self.ranker_data["cat_features"]

        self.df       = self.content_model["df"]
        self.features = self.content_model["features"]
        self.nn       = self.content_model["nn"]

        self.user_vector_builder = UserVectorBuilder(self.content_model)
        self.generator           = CandidateGenerator(self.content_model)

    # ==================================================
    # DIVERSIFICATION
    # ==================================================
    def diversify(self, df, lambda_div=0.02):

        final = []
        for _, row in df.iterrows():
            penalty = sum(
                1 for x in final
                if x["city"] == row["city"]
                or x["property_type"] == row["property_type"]
            )
            penalty = min(penalty, 3)
            score = (
                0.4 * row["similarity"] +
                0.6 * row["rank_score"]
            ) - (lambda_div * penalty)
            score = max(score, 0.0)
            row = row.to_dict()
            row["final_score"] = float(score)
            final.append(row)

        result = pd.DataFrame(final)
        fs = result["final_score"]
        if fs.max() != fs.min():
            result["final_score"] = (fs - fs.min()) / (fs.max() - fs.min() + 1e-9)
        else:
            result["final_score"] = 0.5
        return result.sort_values("final_score", ascending=False)

    # ==================================================
    # COLD START
    # ==================================================
    def cold_start(self, top_n):
        try:
            pop = cache.get_interactions()["property_id"].value_counts()
        except Exception:
            pop = pd.Series(dtype=int)
        df = self.df.copy()
        df["pop_score"] = df["property_id"].map(pop).fillna(0)
        return (
            df.sort_values("pop_score", ascending=False)
              .head(top_n)
              .to_dict("records")
        )

    # ==================================================
    # CONTENT ONLY (weak user)
    # ==================================================
    def content_only(self, user_id, top_n):
        user_profile = self.user_vector_builder.build_user_vector(user_id)
        if user_profile is None:
            return self.cold_start(top_n)

        candidates = self.generator.generate(user_profile["vector"], top_k=200)
        if candidates.empty:
            return self.cold_start(top_n)

        candidates["similarity_score"] = candidates["similarity"]
        return (
            candidates.sort_values("similarity_score", ascending=False)
                       .head(top_n)
                       .to_dict("records")
        )

    # ==================================================
    # FULL HYBRID
    # ==================================================
    def full_hybrid(self, user_id, top_n):
        interactions = cache.get_interactions()
        interactions = interactions[interactions["user_id"] == user_id]

        user_profile = self.user_vector_builder.build_user_vector(user_id)
        if user_profile is None:
            return self.cold_start(top_n)

        candidates = self.generator.generate(user_profile["vector"], top_k=500)
        if candidates.empty:
            return self.cold_start(top_n)

        candidates = candidates.copy()
        candidates["similarity_score"] = candidates["similarity"]

        avg_price = user_profile["avg_price"]
        if avg_price and avg_price > 0:
            candidates["price_affinity"] = np.exp(
                -abs(candidates["price"] - avg_price) / avg_price
            )
        else:
            candidates["price_affinity"] = 0.5

        popularity = cache.get_interactions().groupby("property_id").size()
        candidates["popularity"] = candidates["property_id"].map(popularity).fillna(0)
        candidates["popularity"] /= (candidates["popularity"].max() + 1e-9)

        if "created_at" in candidates.columns:
            days_old = (
                pd.Timestamp.now() -
                pd.to_datetime(candidates["created_at"])
            ).dt.days
            candidates["recency_score"] = np.exp(-days_old / 30)
        else:
            candidates["recency_score"] = 0.5

        candidates["city_match"]        = (candidates["city"]          == user_profile["fav_city"]).astype(int)
        candidates["type_match"]        = (candidates["property_type"] == user_profile["fav_type"]).astype(int)
        candidates["recent_city_match"] = (candidates["city"]          == user_profile["recent_city"]).astype(int)

        recent_session = cache.get_user_session(user_id)
        recent_ids     = [x["property_id"] for x in recent_session]
        candidates["session_recency"] = candidates["property_id"].isin(recent_ids).astype(int)

        X_num = self.scaler.transform(candidates[self.num_features].fillna(0))
        X_cat = self.encoder.transform(candidates[self.cat_features].fillna("unknown"))
        X     = hstack([X_num, X_cat])

        scores  = self.ranker.predict(X)
        scores += candidates["recent_city_match"].values * 0.15
        scores += candidates["session_recency"].values   * 0.10

        if scores.max() != scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.ones_like(scores) * 0.5

        candidates["rank_score"] = scores

        seen_ids   = set(interactions["property_id"].unique())
        candidates = candidates[~candidates["property_id"].isin(seen_ids)]
        candidates = self.diversify(candidates)

        return candidates.head(top_n).to_dict("records")

    # ==================================================
    # MAIN ENTRY
    # ==================================================
    def recommend(self, user_id, top_n=10):
        try:
            interactions = cache.get_interactions()

            # type safety — الـ user_id لازم يتطابق مع نوع الـ DB column
            uid = int(user_id)
            interactions["user_id"] = interactions["user_id"].astype(int)

            user_interactions = interactions[interactions["user_id"] == uid]
            interaction_count = len(user_interactions)

            print(f"🔎 total interactions in cache: {len(interactions)} | user {uid} interactions: {interaction_count}")
        except Exception as e:
            print(f"❌ recommend error: {e}")
            interaction_count = 0

        print(f"🧠 SmartRecommender: user={user_id} interactions={interaction_count}")

        if interaction_count == 0:
            print("→ cold_start")
            return {"strategy": "cold_start", "interaction_count": 0, "results": self.cold_start(top_n)}
        if interaction_count < 5:
            print("→ content_only")
            return {"strategy": "content_only", "interaction_count": interaction_count, "results": self.content_only(user_id, top_n)}
        print("→ full_hybrid")
        return {"strategy": "full_hybrid", "interaction_count": interaction_count, "results": self.full_hybrid(user_id, top_n)}