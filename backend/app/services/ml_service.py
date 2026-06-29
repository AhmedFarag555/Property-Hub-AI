import os
import sys

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
ML_DIR = os.path.join(BASE_DIR, "ml")

for p in [BASE_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


class RecommendationService:

    def __init__(self):
        self._smart   = None
        self._content = None

    def _get_smart(self):
        if self._smart is None:
            from ml.smart_recommender import SmartRecommender
            from ml.data_cache import cache
            self._smart = SmartRecommender(cache=cache)
        return self._smart

    def _get_content(self):
        if self._content is None:
            from ml.content_model import ContentBasedRecommender
            self._content = ContentBasedRecommender(
                os.path.join(ML_DIR, "model.pkl")
            )
        return self._content

    def smart_recommend(self, user_id, top_n=10):
        output = self._get_smart().recommend(user_id=user_id, top_n=top_n)
        # SmartRecommender بيرجع dict فيه strategy + results
        if isinstance(output, dict):
            return output.get("results", [])
        return output or []

    def cold_start(self, top_n=10):
        return self._get_smart().cold_start(top_n=top_n)