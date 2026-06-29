import ast
import json
import re
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import os, sys
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)
from custom_transformers import LocationClusterer




class PricePredictionModel:
    def __init__(self, model_path: str = "price_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.metrics: Dict[str, Any] = {}
        self.best_params: Dict[str, Any] = {}

    @staticmethod
    def _parse_amenities(value: Any) -> list:
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(value)
                        return parsed if isinstance(parsed, list) else []
                    except Exception:
                        continue
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @staticmethod
    def _remove_price_leakage(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"\b(price|rent|asking|monthly|per month|down payment|installment)s?\b[^.،,\n]*", " ", text)
        text = re.sub(r"\b\d+(?:[,.]\d+)?\s*(egp|usd|جنيه|دولار|million|thousand|k|m)\b", " ", text)
        text = re.sub(r"\b(egp|e\.g\.p|usd|dollar|pound|جنيه|دولار)\b", " ", text)
        text = re.sub(r"\b\d{4,}\b", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def prepare_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        text_cols = ["title", "description", "location_full", "compound","area", "city", "property_type", "offering_type"]
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str).str.lower().str.strip()

        numeric_cols = ["area_sqm", "bedrooms", "bathrooms", "latitude", "longitude", "amenities_count"]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "amenities" not in df.columns:
            df["amenities"] = [[] for _ in range(len(df))]

        df["amenities"] = df["amenities"].apply(cls._parse_amenities)
        df["amenities_count"] = df["amenities"].apply(len)
        df["amenities_text"] = df["amenities"].apply(
            lambda items: " ".join(str(item).lower().strip() for item in items)
        )
        
        raw_text = df["title"] + " " + df["description"] + " " + df["location_full"] + " " + df["amenities_text"]
        df["search_text"] = raw_text.apply(cls._remove_price_leakage)

        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        before = len(df)

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce")
        df = df.dropna(subset=["price", "area_sqm"])
        df = df[(df["price"] > 0) & (df["area_sqm"] > 0)]

        group_cols = [col for col in ["offering_type", "property_type"] if col in df.columns]
        if group_cols:
            grouped = df.groupby(group_cols, dropna=False)["price"]
            low = grouped.transform(lambda s: s.quantile(0.01))
            high = grouped.transform(lambda s: s.quantile(0.99))
            df = df[(df["price"] >= low) & (df["price"] <= high)]
        else:
            df = df[df["price"].between(df["price"].quantile(0.01), df["price"].quantile(0.99))]

        df = df.copy()
        df.loc[:, "price_per_sqm_tmp"] = df["price"] / (df["area_sqm"] + 1e-9)
        if group_cols:
            grouped_ppm = df.groupby(group_cols, dropna=False)["price_per_sqm_tmp"]
            low_ppm = grouped_ppm.transform(lambda s: s.quantile(0.01))
            high_ppm = grouped_ppm.transform(lambda s: s.quantile(0.99))
            df = df[(df["price_per_sqm_tmp"] >= low_ppm) & (df["price_per_sqm_tmp"] <= high_ppm)]
        else:
            df = df[df["price_per_sqm_tmp"].between(df["price_per_sqm_tmp"].quantile(0.01), df["price_per_sqm_tmp"].quantile(0.99))]

        df = df.drop(columns=["price_per_sqm_tmp"])
        df.attrs["removed_outliers"] = before - len(df)
        return df.reset_index(drop=True)

    @staticmethod
    def _feature_pipeline() -> Pipeline:
        numeric_features = ["area_sqm", "bedrooms", "bathrooms", "latitude", "longitude", "amenities_count"]
        categorical_features = ["property_type", "city","area", "compound", "offering_type", "location_cluster"]

        return Pipeline(
            steps=[
                ("location_cluster", LocationClusterer(n_clusters=50)),
                (
                    "columns",
                    ColumnTransformer(
                        transformers=[
                            ("num", SimpleImputer(strategy="median"), numeric_features),
                            (
                                "cat",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                    ]
                                ),
                                categorical_features,
                            ),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    max_features=5000,
                                    min_df=2,
                                    ngram_range=(1, 2),
                                    sublinear_tf=True,
                                    strip_accents="unicode",
                                ),
                                "search_text",
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
            ]
        )

    @staticmethod
    def _split_data(X: pd.DataFrame, y: pd.Series, test_size: float):
        for group_col in ("listing_id", "property_id"):
            if group_col in X.columns and X[group_col].dropna().nunique() > 1:
                splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                train_idx, test_idx = next(splitter.split(X, y, groups=X[group_col].fillna("missing")))
                return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], f"group:{group_col}"

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=X["offering_type"] if X["offering_type"].nunique() > 1 else None,
        )
        return X_train, X_test, y_train, y_test, "random"

    def _model(self) -> TransformedTargetRegressor:
        pipeline = Pipeline(
            steps=[
                ("preprocess", self._feature_pipeline()),
                (
                    "model",
                    LGBMRegressor(
                        objective="regression_l1",
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )

        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    @staticmethod
    def _param_distributions() -> Dict[str, Any]:
        return {
            "regressor__preprocess__location_cluster__n_clusters": [30, 50, 70],
            "regressor__model__n_estimators": randint(300, 900),
            "regressor__model__learning_rate": uniform(0.03, 0.07),
            "regressor__model__num_leaves": randint(24, 96),
            "regressor__model__max_depth": [-1, 8, 12, 16],
            "regressor__model__min_child_samples": randint(20, 90),
            "regressor__model__subsample": uniform(0.75, 0.25),
            "regressor__model__colsample_bytree": uniform(0.7, 0.3),
            "regressor__model__reg_alpha": uniform(0.0, 0.5),
            "regressor__model__reg_lambda": uniform(0.0, 1.0),
        }

    def train(
        self,
        df: pd.DataFrame,
        n_iter: int = 12,
        test_size: float = 0.2,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.prepare_dataframe(df)
        df = self.remove_outliers(df)
        removed_outliers = int(df.attrs.get("removed_outliers", 0))

        y = df["price"].astype(float)
        X = df.drop(columns=["price"])
        X_train, X_test, y_train, y_test, split_strategy = self._split_data(X, y, test_size)

        search = RandomizedSearchCV(
            estimator=self._model(),
            param_distributions=self._param_distributions(),
            n_iter=n_iter,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)

        self.pipeline = search.best_estimator_
        predictions = np.maximum(self.pipeline.predict(X_test), 0)
        actual = y_test

        self.metrics = {
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "r2": float(r2_score(actual, predictions)),
            "mape": float(np.mean(np.abs((actual - predictions) / np.maximum(actual, 1))) * 100),
            "split_strategy": split_strategy,
            "removed_outliers": removed_outliers,
            "target_transform": "log1p",
            "tfidf_max_features": 5000,
            "tfidf_ngram_range": "(1, 2)",
            "model": "LightGBM",
            "location_clustering": True,
        }
        self.best_params = search.best_params_

        output_path = save_path or self.model_path
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "metrics": self.metrics,
                "best_params": self.best_params,
                "trained_rows": int(len(df)),
            },
            output_path,
        )

        return {
            "metrics": self.metrics,
            "best_params": self.best_params,
            "trained_rows": int(len(df)),
            "model_path": output_path,
        }

    def load(self, model_path: Optional[str] = None) -> "PricePredictionModel":
        data = joblib.load(model_path or self.model_path)
        self.pipeline = data["pipeline"]
        self.metrics = data.get("metrics", {})
        self.best_params = data.get("best_params", {})
        return self

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipeline is None:
            self.load()

        df = self.prepare_dataframe(pd.DataFrame([payload]))
        predicted_price = float(max(self.pipeline.predict(df)[0], 0))

        return {
            "predicted_price": round(predicted_price, 2),
            "currency": payload.get("currency", "EGP"),
            "metrics": self.metrics,
        }
        
        
if __name__ == "__main__":

    from data_cache import cache

    print("Loading data from cache...")

    df = cache.properties.copy()
    

    print("Rows:", len(df))

    model = PricePredictionModel()

    print("Training model...")

    results = model.train(df=df, n_iter=5, save_path="ml\price_model.pkl")

    print("\n===== TRAINING RESULTS =====\n")

    for k, v in results["metrics"].items():
        print(f"{k}: {v}")

    print("\n===== BEST PARAMS =====\n")

    for k, v in results["best_params"].items():
        print(f"{k}: {v}")

    print("\nModel saved successfully ✅")
