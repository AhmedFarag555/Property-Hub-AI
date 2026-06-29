from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app.core.deps import get_current_user
from app.models.user import User
from app.models.user_behavior import UserBehavior

router = APIRouter(prefix="/profile", tags=["Profile"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/interactions")
def get_user_interactions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """✅ جيب كل interactions بتاعت اليوزر ده بس — view/favorite/click/contact"""
    from sqlalchemy import text as _text
    rows = db.execute(
        _text("""
            SELECT property_id, action, created_at
            FROM interactions
            WHERE user_id = :uid
            ORDER BY created_at DESC
        """),
        {"uid": current_user.user_id}
    ).fetchall()
    return [
        {"property_id": r[0], "action": r[1], "created_at": str(r[2])}
        for r in rows
    ]


@router.get("/behavior")
def get_user_behavior(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """جيب كل الـ user_behavior records بتاعت اليوزر"""
    records = (
        db.query(UserBehavior)
        .filter(UserBehavior.user_id == current_user.user_id)
        .order_by(UserBehavior.last_seen.desc())
        .all()
    )
    return [
        {
            "property_id":   r.property_id,
            "price":         r.price,
            "bedrooms":      r.bedrooms,
            "bathrooms":     r.bathrooms,
            "area_sqm":      r.area_sqm,
            "city":          r.city,
            "compound":      r.compound,
            "area":          r.area,
            "property_type": r.property_type,
            "offering_type": r.offering_type,
            "interest_score": r.interest_score,
            "last_seen":     str(r.last_seen) if r.last_seen else None,
        }
        for r in records
    ]