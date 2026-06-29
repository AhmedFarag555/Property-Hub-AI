from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd



class LocationClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 120, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None

    def fit(self, X, y=None):
        coords = self._coords(X)
        unique_points = np.unique(coords, axis=0)
        n_clusters = min(self.n_clusters, max(len(unique_points), 1))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans.fit(coords)
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        coords = self._coords(df)
        df["location_cluster"] = self.kmeans.predict(coords).astype(str)
        return df

    @staticmethod
    def _coords(X):
        df = pd.DataFrame(X).copy()
        lat = pd.to_numeric(df.get("latitude", 0), errors="coerce").fillna(0)
        lon = pd.to_numeric(df.get("longitude", 0), errors="coerce").fillna(0)
        return np.column_stack([lat, lon])