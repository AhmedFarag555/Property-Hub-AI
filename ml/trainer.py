import joblib
from sklearn.neighbors import NearestNeighbors

from data_loader import DataLoader
from feature_builder import FeatureBuilder


class Trainer:

    def __init__(self, db_url):
        self.loader = DataLoader(db_url)
        self.builder = FeatureBuilder()

    def train_and_save(self):

        # ---------------- load data
        df = self.loader.load_data()

        # ---------------- build features
        self.builder.fit(df)
        features = self.builder.transform(df)
        
        from sklearn.preprocessing import normalize
        features = normalize(features)

        # ---------------- train model
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(features)

        # ---------------- SAVE EVERYTHING
        joblib.dump({
            "df": df,
            "features": features,
            "nn": nn,
            "feature_builder": self.builder
        }, "ml\model.pkl")

        print(" Model saved successfully")


if __name__ == "__main__":

    trainer = Trainer(
        "mysql+mysqlconnector://root:1234@localhost/real_estate_db"
    )

    trainer.train_and_save()