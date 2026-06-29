from fastapi import APIRouter, Depends
from typing import Optional

from app.core.deps import get_current_user
from app.models.user import User

from app.services.ml_service import RecommendationService

router = APIRouter(prefix="/ml", tags=["ML"])

rec_service = RecommendationService()


@router.get("/recommend")
def recommend(
    top_n: int = 10,
    property_id: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):

    user_id = current_user.user_id

    # =====================
    # SMART FIRST
    # =====================
    results = rec_service.smart_recommend(
        user_id=user_id,
        top_n=top_n
    )

    # =====================
    # COLD START FALLBACK
    # =====================
    if not results and property_id:

        fallback = rec_service.cold_start(
            property_id=property_id,
            top_n=top_n
        )

        return {
            "type": "content-based-cold-start",
            "results": fallback
        }

    return {
        "type": "hybrid",
        "results": results
    }

    