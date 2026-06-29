import numpy as np

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from sklearn.preprocessing import normalize

from user_vector_builder import UserVectorBuilder


class RecommendationService:

    def __init__(self):

        self.builder = UserVectorBuilder()

        self.df = self.builder.df

        self.features = self.builder.features

    # ------------------------------------------------
    # RECOMMEND
    # ------------------------------------------------
    def recommend(self, user_id, top_n=10):

        user_vector = (
            self.builder
            .build_user_vector(user_id)
        )

        # cold user
        if user_vector is None:

            return self.df.sample(
                min(top_n, len(self.df))
            ).to_dict(orient="records")

        user_vector = normalize(user_vector)

        similarities = cosine_similarity(
            user_vector,
            self.features
        )[0]

        df = self.df.copy()

        df["score"] = similarities

        # remove already seen
        from data_cache import cache

        seen = cache.interactions[
            cache.interactions["user_id"]
            == user_id
        ]["property_id"].unique()

        df = df[
            ~df["property_id"].isin(seen)
        ]

        df = df.sort_values(
            "score",
            ascending=False
        )

        return df.head(top_n).to_dict(
            orient="records"
        )