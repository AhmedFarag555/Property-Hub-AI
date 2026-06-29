import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


class FeatureBuilder:

    def fit(self, df):

        df = df.copy()

        # ---------------- clean missing ----------------
        df["property_type"] = df["property_type"].fillna("unknown")
        df["city"] = df["city"].fillna("unknown")
        df["offering_type"] = df["offering_type"].fillna("unknown")
        df["area"] = df["area"].fillna("unknown")
        # ---------------- amenities clean ----------------
        df["amenities"] = df["amenities"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        # ---------------- categorical ----------------
        self.ohe = OneHotEncoder(handle_unknown="ignore")

        self.ohe.fit(df[[
            "property_type",
            "city",
            "area",
            "offering_type"
        ]])

        # ---------------- numeric ----------------
        self.scaler = MinMaxScaler()

        num_cols = [
            "price", "area_sqm", "bedrooms",
            "bathrooms", "price_per_sqm", "amenities_count"
        ]

        self.scaler.fit(df[num_cols].fillna(0))

        # ---------------- text ----------------
        df["text"] = self._build_text(df)

        self.tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words=None
        )

        self.tfidf.fit(df["text"])

    # ---------------- transform ----------------
    def transform(self, df):

        df = df.copy()

        df["property_type"] = df["property_type"].fillna("unknown")
        df["city"] = df["city"].fillna("unknown")
        df["area"] = df["area"].fillna("unknown")
        df["offering_type"] = df["offering_type"].fillna("unknown")

        df["amenities"] = df["amenities"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        df["text"] = self._build_text(df)

        # categorical
        cat = self.ohe.transform(df[[
            "property_type",
            "city",
            "area",
            "offering_type"
        ]])

        # numeric
        num = self.scaler.transform(df[[
            "price", "area_sqm", "bedrooms",
            "bathrooms", "price_per_sqm", "amenities_count"
        ]].fillna(0))

        # text
        text = self.tfidf.transform(df["text"])

        # ---------------- weighted merge ----------------
        features = hstack([
            num * 1.0,
            cat * 1.5,
            text * 2.0
        ]).tocsr()

        return features

    # ---------------- text builder ----------------
    def _build_text(self, df):

        return (
            df["title"].fillna("") + " " +
            df["description"].fillna("") + " " +
            df["property_type"].fillna("") + " " +
            df["city"].fillna("") + " " +
            df["area"].fillna("") + " " +
            df["offering_type"].fillna("") + " " +
            df["amenities"].apply(
                lambda x: " ".join(x) if isinstance(x, list) else ""
            )
        )
        