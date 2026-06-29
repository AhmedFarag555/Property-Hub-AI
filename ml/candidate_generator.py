import numpy as np


class CandidateGenerator:

    def __init__(self, model_data):

        self.df = model_data["df"]
        self.nn = model_data["nn"]

    def generate(self, user_vector, top_k=500):

        if user_vector is None:
            return []

        user_vector = np.asarray(user_vector).reshape(1, -1)

        distances, indices = self.nn.kneighbors(
            user_vector,
            n_neighbors=top_k
        )

        candidates = self.df.iloc[indices[0]].copy()

        candidates["similarity"] = 1 - distances[0]

        return candidates.sort_values(
            "similarity",
            ascending=False
        )