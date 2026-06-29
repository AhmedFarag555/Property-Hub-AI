import pandas as pd
import json
from sqlalchemy import create_engine


class DataLoader:

    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def load_data(self):

        df = pd.read_sql("SELECT * FROM properties_clean", self.engine)

        df["property_id"] = pd.to_numeric(df["property_id"], errors="coerce")
        df = df.dropna(subset=["property_id"])
        df["property_id"] = df["property_id"].astype(int)

        df["amenities"] = df["amenities"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

        df["compound"] = df["compound"].fillna("unknown")
        df["city"] = df["city"].fillna("unknown")
        df["area"] = df["area"].fillna("unknown")
        df["property_type"] = df["property_type"].fillna("unknown")

        df = df.dropna(subset=["latitude", "longitude"])

        return df.reset_index(drop=True)