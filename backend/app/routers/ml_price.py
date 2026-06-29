import sys
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
ML_DIR = os.path.join(BASE_DIR, "ml")

# ── Add both to path so ml/ internal imports work too ──
for p in [BASE_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ml.content_model import ContentBasedRecommender
from ml.price_model import PricePredictionModel
from ml.smart_recommender import SmartRecommender
from app.core.deps import get_current_user, get_optional_user
from app.models.user import User
from app.database.database import SessionLocal
from app.models.interaction import Interaction
from sqlalchemy.orm import Session
from datetime import datetime


# ─────────────────────────────────────────
# LOAD MODELS (once at startup)
# ─────────────────────────────────────────
MODEL_PATH         = os.path.join(ML_DIR, "price_model.pkl")
CONTENT_MODEL_PATH = os.path.join(ML_DIR, "model.pkl")

price_model   = PricePredictionModel()
price_model.load(MODEL_PATH)

content_model = ContentBasedRecommender(CONTENT_MODEL_PATH)

# ── SmartRecommender — تحميل الـ models مرة واحدة، قراءة الـ cache كل request ──
_smart_instance = None

def get_smart_recommender():
    global _smart_instance
    if _smart_instance is None:
        from ml.smart_recommender import SmartRecommender
        _smart_instance = SmartRecommender()
        print("✅ SmartRecommender loaded")
    return _smart_instance

def reset_smart_recommender():
    pass


# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────
router = APIRouter(prefix="/ml", tags=["ML"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
class PropertyInput(BaseModel):
    city:          str
    area:          Optional[str] = ""
    compound:      Optional[str] = ""
    property_type: str
    offering_type: str
    area_sqm:      float
    bedrooms:      int
    bathrooms:     int
    amenities:     Optional[List[str]] = []
    description:   Optional[str] = ""


class TrackRequest(BaseModel):
    property_id: int
    action:      str   # view | click | favorite | unfavorite | contact


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _build_price_payload(data: PropertyInput) -> dict:
    return {
        "city":          data.city,
        "area":          data.area,
        "compound":      data.compound,
        "property_type": data.property_type,
        "offering_type": data.offering_type,
        "area_sqm":      data.area_sqm,
        "bedrooms":      data.bedrooms,
        "bathrooms":     data.bathrooms,
        "amenities":     data.amenities,
        "title":         "",
        "location_full": f"{data.city} {data.area}",
        "latitude":      0,
        "longitude":     0,
        "description":   data.description,
    }


def _clean_rec(r: dict) -> dict:
    """Make a recommendation JSON-serialisable."""
    prop = r.get("property", {})
    clean = {}
    for k, v in prop.items():
        if hasattr(v, "item"):          # numpy scalar
            v = v.item()
        elif hasattr(v, "tolist"):      # numpy array
            v = v.tolist()
        clean[k] = v
    return {"property": clean, "score": round(float(r.get("content_score", 0)), 4)}


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@router.post("/predict")
def predict_price(data: PropertyInput):
    """Price prediction only."""
    result = price_model.predict(_build_price_payload(data))
    return {
        "price_prediction": result["predicted_price"],
        "currency":         result.get("currency", "EGP"),
    }


@router.post("/price-with-suggestions")
def price_with_suggestions(data: PropertyInput):
    """Price prediction + similar property recommendations."""
    price_result = price_model.predict(_build_price_payload(data))

    recommendations = content_model.recommend_from_features(
        city=data.city,
        area=data.area,
        compound=data.compound,
        property_type=data.property_type,
        offering_type=data.offering_type,
        area_sqm=data.area_sqm,
        bedrooms=data.bedrooms,
        bathrooms=data.bathrooms,
        amenities=data.amenities,
        top_n=10,
    )

    return {
        "prediction": {
            "price":    price_result["predicted_price"],
            "currency": price_result.get("currency", "EGP"),
        },
        "recommendations": [_clean_rec(r) for r in recommendations],
    }


@router.get("/recommend")
def recommend(
    top_n: int = 10,
    property_id: Optional[int] = None,
    current_user: User = Depends(get_optional_user),
):
    """
    Smart hybrid recommendations.
    - لو في user مسجّل → hybrid (history + content)
    - لو مفيش → content-based cold start بالـ property_id
    """

    # ── HYBRID (logged in) ──
    if current_user:
        try:
            # cache دايماً محدّث real-time عن طريق add_interaction
            smart = get_smart_recommender()

            rec_output = smart.recommend(user_id=current_user.user_id, top_n=top_n)

            # smart.recommend بيرجع dict فيه strategy + interaction_count + results
            if isinstance(rec_output, dict):
                strategy          = rec_output.get("strategy", "unknown")
                interaction_count = rec_output.get("interaction_count", 0)
                raw               = rec_output.get("results", [])
            else:
                strategy          = "hybrid"
                interaction_count = 0
                raw               = rec_output or []

            print(f"🔍 user_id={current_user.user_id} | strategy={strategy} | interactions={interaction_count} | results={len(raw)}")

            # clean numpy types دايماً حتى لو raw فاضي
            cleaned = []
            for r in (raw or []):
                if isinstance(r, dict):
                    clean = {}
                    for k, v in r.items():
                        if hasattr(v, "item"):     v = v.item()
                        elif hasattr(v, "tolist"): v = v.tolist()
                        clean[k] = v
                    cleaned.append(clean)
                else:
                    cleaned.append(r)

            # لو raw فاضي → جرب cold start بالـ property_id
            if not cleaned and property_id:
                fallback = content_model.recommend(property_id=property_id, top_n=top_n)
                cleaned  = [_clean_rec(r) for r in fallback]
                strategy = "content-based-cold-start"

            return {
                "type":               strategy,
                "user_id":            current_user.user_id,
                "interaction_count":  interaction_count,
                "results":            cleaned
            }

        except Exception as e:
            print(f"⚠️  SmartRecommender failed: {e}")
            import traceback; traceback.print_exc()

    # ── COLD START (no user or smart failed) ──
    if property_id:
        fallback = content_model.recommend(property_id=property_id, top_n=top_n)
        return {
            "type":    "content-based-cold-start",
            "user_id": current_user.user_id if current_user else None,
            "results": [_clean_rec(r) for r in fallback]
        }

    # آخر fallback — popular properties
    smart   = get_smart_recommender()
    popular = smart.cold_start(top_n=top_n)
    return {
        "type":               "popular",
        "user_id":            current_user.user_id if current_user else None,
        "interaction_count":  0,
        "results":            popular
    }


@router.post("/track")
def track_action(
    data: TrackRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Track a user interaction (view / click / favorite / contact)."""
    from sqlalchemy import text as _text
    from app.models.user_behavior import UserBehavior

    VALID_ACTIONS = {"view", "click", "favorite", "unfavorite", "contact"}
    if data.action not in VALID_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid action. Use: {VALID_ACTIONS}")

    # ── 1. unfavorite: امسح من interactions وuser_behavior وارجع ────────────
    if data.action == "unfavorite":
        db.execute(
            _text("""
                DELETE FROM interactions
                WHERE user_id = :uid AND property_id = :pid AND action = 'favorite'
            """),
            {"uid": current_user.user_id, "pid": data.property_id}
        )
        db.execute(
            _text("""
                UPDATE user_behavior
                SET interest_score = GREATEST(0, interest_score - 5),
                    last_seen      = :now
                WHERE user_id = :uid AND property_id = :pid
            """),
            {"uid": current_user.user_id, "pid": data.property_id, "now": datetime.utcnow()}
        )
        db.commit()
        return {"status": "success", "message": "Unfavorited"}

    # ── 2. INSERT IGNORE — منع التكرار على مستوى الـ DB مهما حصل في الـ frontend
    #    لو UNIQUE KEY مش موجود → بنعمل SELECT أولاً كـ fallback مضمون
    result = db.execute(
        _text("""
            INSERT IGNORE INTO interactions (user_id, property_id, action, created_at)
            SELECT :uid, :pid, :act, :now
            WHERE NOT EXISTS (
                SELECT 1 FROM interactions
                WHERE user_id    = :uid
                  AND property_id = :pid
                  AND action      = :act
            )
        """),
        {
            "uid": current_user.user_id,
            "pid": data.property_id,
            "act": data.action,
            "now": datetime.utcnow(),
        }
    )
    db.commit()

    # لو مفيش rows اتضافت → سبق اتسجلت → وقف
    if result.rowcount == 0:
        return {"status": "skipped", "message": "Interaction already recorded"}

    # ── 3. جيب بيانات العقار من الـ DB ────────────────────────────────────────
    prop_row = db.execute(
        _text("""
            SELECT price, bedrooms, bathrooms, area_sqm,
                   city, compound, area, property_type, offering_type
            FROM properties
            WHERE property_id = :pid
            LIMIT 1
        """),
        {"pid": data.property_id}
    ).fetchone()

    # ── 4. interest_score بناءً على نوع الـ action ─────────────────────────────
    ACTION_SCORE = {"view": 1.0, "click": 2.0, "favorite": 5.0, "contact": 10.0}
    score_delta  = ACTION_SCORE.get(data.action, 1.0)

    # ── 5. UPSERT في user_behavior ─────────────────────────────────────────────
    #       لو الـ row موجودة → زوّد الـ interest_score وحدّث last_seen
    #       لو مش موجودة → أنشئها
    existing_beh = db.execute(
        _text("""
            SELECT interest_score FROM user_behavior
            WHERE user_id = :uid AND property_id = :pid
            LIMIT 1
        """),
        {"uid": current_user.user_id, "pid": data.property_id}
    ).fetchone()

    if existing_beh:
        new_score = round((existing_beh[0] or 0) + score_delta, 2)
        db.execute(
            _text("""
                UPDATE user_behavior
                SET interest_score = :score,
                    last_seen      = :now
                WHERE user_id = :uid AND property_id = :pid
            """),
            {"score": new_score, "now": datetime.utcnow(),
             "uid": current_user.user_id, "pid": data.property_id}
        )
    else:
        p = prop_row
        db.add(UserBehavior(
            user_id       = current_user.user_id,
            property_id   = data.property_id,
            price         = float(p[0]) if p and p[0] else None,
            bedrooms      = int(p[1])   if p and p[1] else None,
            bathrooms     = int(p[2])   if p and p[2] else None,
            area_sqm      = float(p[3]) if p and p[3] else None,
            city          = p[4]        if p else None,
            compound      = p[5]        if p else None,
            area          = p[6]        if p else None,
            property_type = p[7]        if p else None,
            offering_type = p[8]        if p else None,
            interest_score = score_delta,
            last_seen      = datetime.utcnow(),
        ))

    db.commit()

    # ⚡ real-time update — بنستخدم نفس الـ cache instance اللي الـ ML بيستخدمه
    try:
        import sys

        # جيب الـ cache من الـ ml module مباشرة عشان يكون نفس الـ instance
        if "data_cache" in sys.modules:
            cache_module = sys.modules["data_cache"]
        elif "ml.data_cache" in sys.modules:
            cache_module = sys.modules["ml.data_cache"]
        else:
            import importlib, os
            spec = importlib.util.spec_from_file_location(
                "data_cache",
                os.path.join(ML_DIR, "data_cache.py")
            )
            cache_module = importlib.util.module_from_spec(spec)
            sys.modules["data_cache"] = cache_module
            spec.loader.exec_module(cache_module)

        cache_module.cache.add_interaction(
            user_id=int(current_user.user_id),
            property_id=int(data.property_id),
            action=data.action
        )
        print(f"⚡ interaction added via shared cache | pending={len(cache_module.cache._pending_interactions)}")
    except Exception as e:
        print(f"⚠️ cache.add_interaction failed: {e}")
        import traceback; traceback.print_exc()

    return {"status": "success", "message": "Interaction tracked"}