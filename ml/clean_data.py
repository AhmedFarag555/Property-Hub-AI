import pandas as pd
from sqlalchemy import create_engine
import ast
import json

class DataPipeline:

    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    # ----------------------------
    #  Load Data
    # ----------------------------
    def load_data(self):
        df = pd.read_sql("SELECT * FROM properties", self.engine)
        print(" Loaded:", df.shape)
        return df

    # ----------------------------
    #  Clean IDs
    # ----------------------------
    def clean_ids(self, df):
        df["property_id"] = pd.to_numeric(df["property_id"], errors="coerce")
        df = df.dropna(subset=["property_id","all_image_url"])
        df = df[df['all_image_url'] != '[]']
        df["property_id"] = df["property_id"].astype(int)
        df = df.drop_duplicates(subset=["property_id"])
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        return df

    # ----------------------------
    # Fix numeric columns
    # ----------------------------
    def fix_numeric(self, df):
        cols = ["price", "area_sqm", "bedrooms", "bathrooms", "latitude", "longitude"]

        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ----------------------------
    #  Remove invalid rows
    # ----------------------------
    def remove_invalid(self, df):
        df = df.dropna(subset=["price", "area_sqm", "bedrooms", "bathrooms", "city"])
        df = df[(df["price"] > 0) & (df["area_sqm"] > 0)]
        return df

    # ----------------------------
    #  Clean text
    # ----------------------------
    def clean_text(self, df):
        df["city"] = df["city"].str.lower().str.strip()
        df["property_type"] = df["property_type"].str.lower().str.strip()
        df["offering_type"] = df["offering_type"].str.lower().str.strip()
        df["compound"] = df["compound"].fillna("").str.lower().str.strip()
        df["area"] = df["area"].fillna("").str.lower().str.strip()
        return df

    # ----------------------------
    #  Amenities
    # ----------------------------
    def clean_amenities(self, df):

        def parse(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                if x.startswith("["):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return []
                return [i.strip().lower() for i in x.split(",") if i.strip()]
            return []

        df["amenities"] = df["amenities"].apply(parse)
        return df

    # ----------------------------
    #  Feature Engineering
    # ----------------------------
    def feature_engineering(self, df):
        df["price_per_sqm"] = df["price"] / df["area_sqm"]
        df["amenities_count"] = df["amenities"].apply(len)
        return df

    # ----------------------------
    #  Drop unused columns
    # ----------------------------
    def drop_columns(self, df):
        drop_cols = [
            "share_url",
            "price_per_sqm_display",
            "price_period",
            "image_urls"
        ]

        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df

    # ----------------------------
    #  Save Data
    # ----------------------------
    def save_data(self, df):

        
        df["amenities"] = df["amenities"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )



    
        if "index" in df.columns:
            df = df.drop(columns=["index"])

        df.to_sql(
            name="properties_clean",
            con=self.engine,
            if_exists="replace",
            chunksize=1000,
            index=False
        )

        print("Clean data saved successfully!")
    
    # ----------------------------
    # Run Full Pipeline
    # ----------------------------
    def run(self):
        df = self.load_data()
        df = self.clean_ids(df)
        df = self.fix_numeric(df)
        df = self.remove_invalid(df)
        df = self.clean_text(df)
        df = self.clean_amenities(df)
        df = self.feature_engineering(df)
        df = self.drop_columns(df)

        df = df.reset_index(drop=True)

        print("Final shape:", df.shape)

        self.save_data(df)

        return df
    


if __name__ == "__main__":
    pipeline = DataPipeline(
        "mysql+mysqlconnector://root:1234@localhost/real_estate_db"
    )

    pipeline.run()