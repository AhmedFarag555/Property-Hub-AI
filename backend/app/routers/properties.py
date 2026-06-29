from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import sys, os, importlib, importlib.util
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
ML_DIR   = os.path.join(BASE_DIR, "ml")
for p in [BASE_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

def get_cache():
    if "data_cache" in sys.modules:
        return sys.modules["data_cache"].cache
    elif "ml.data_cache" in sys.modules:
        return sys.modules["ml.data_cache"].cache
    else:
        spec = importlib.util.spec_from_file_location(
            "data_cache", os.path.join(ML_DIR, "data_cache.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["data_cache"] = mod
        spec.loader.exec_module(mod)
        return mod.cache

from app.database.database import SessionLocal
from app.core.deps import get_current_user, get_optional_user
from app.models.user import User

router = APIRouter(prefix="/properties", tags=["Properties"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_df():
    return get_cache().properties

def get_filters(df):
    return {
        "property_type":  df["property_type"].dropna().unique().tolist(),
        "city":           df["city"].dropna().unique().tolist(),
        "area":           df["area"].dropna().unique().tolist(),
        "price_min":      float(df["price"].min()),
        "price_max":      float(df["price"].max()),
        "bedrooms_min":   int(df["bedrooms"].min()),
        "bedrooms_max":   int(df["bedrooms"].max()),
        "bathrooms_min":  int(df["bathrooms"].min()),
        "bathrooms_max":  int(df["bathrooms"].max()),
    }

# ── GET / — للموقع العام من الـ cache ─────────────────────────────────────────
@router.get("/")
def get_all_properties(
    page: int = Query(1, ge=1),
    limit: int = Query(30, ge=1),
    property_type: str = None,
    search: str = None,
    min_price: float = None,
    max_price: float = None,
    bedrooms: int = None,
    bathrooms: int = None,
    offering_type: str = None,
    city: str = None
):
    df = get_df()
    filters = get_filters(df)

    if search:
        s = search.lower()
        df = df[
            df["title"].str.lower().str.contains(s, na=False) |
            df["city"].str.lower().str.contains(s, na=False) |
            df["area"].str.lower().str.contains(s, na=False) |
            df["property_type"].str.lower().str.contains(s, na=False)
        ]
    if property_type:
        df = df[df["property_type"] == property_type]
    if city:
        df = df[df["city"].str.lower().str.contains(city.lower(), na=False)]
    if offering_type:
        df = df[
            (df["offering_type"] == offering_type) |
            (df["offering_type"].isna()) |
            (df["offering_type"] == "")
        ]
    if min_price is not None:
        df = df[(df["price"] >= min_price) | (df["price"].isna())]
    if max_price is not None:
        df = df[(df["price"] <= max_price) | (df["price"].isna())]
    if bedrooms is not None:
        df = df[(df["bedrooms"] >= bedrooms) | (df["bedrooms"].isna())]
    if bathrooms is not None:
        df = df[(df["bathrooms"] >= bathrooms) | (df["bathrooms"].isna())]

    total = len(df)

    if "updated_at" in df.columns:
        df = df.sort_values("updated_at", ascending=False, na_position="last")

    start = (page - 1) * limit
    paginated_df = df.iloc[start:start + limit]

    return {
        "page": page, "limit": limit, "total": total,
        "total_pages": (total // limit) + 1,
        "properties": [
            {k: (None if (isinstance(v, float) and v != v) else v)
             for k, v in row.items()}
            for row in paginated_df.to_dict(orient="records")
        ],
        "filters": filters
    }

# ── GET /admin/all — للـ manage page بـ pagination + search + property_id ──────
@router.get("/admin/all")
def get_admin_properties(
    page:          int = Query(1, ge=1),
    limit:         int = Query(50, ge=1, le=100),
    search:        str = None,
    property_id:   int = None,
    offering_type: str = None,
    property_type: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    from app.models.property import Property as PropertyModel
    from sqlalchemy import or_

    q = db.query(PropertyModel)

    if property_id:
        q = q.filter(PropertyModel.property_id == property_id)
    else:
        if search:
            s = f"%{search}%"
            q = q.filter(or_(
                PropertyModel.title.ilike(s),
                PropertyModel.city.ilike(s),
                PropertyModel.area.ilike(s),
                PropertyModel.compound.ilike(s),
            ))
        if offering_type:
            q = q.filter(PropertyModel.offering_type == offering_type)
        if property_type:
            q = q.filter(PropertyModel.property_type == property_type)

    total = q.count()
    props = q.order_by(PropertyModel.property_id.desc()) \
             .offset((page - 1) * limit).limit(limit).all()

    return {
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": max(1, (total + limit - 1) // limit),
        "properties": [
            {c.name: getattr(p, c.name) for c in PropertyModel.__table__.columns}
            for p in props
        ]
    }

# ── GET /admin/filters — قيم الفلاتر للـ dropdowns ──────────────────────────
@router.get("/admin/filters")
def get_admin_filters(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    from app.models.property import Property as PropertyModel
    from sqlalchemy import distinct

    offerings = [r[0] for r in db.query(distinct(PropertyModel.offering_type)).filter(PropertyModel.offering_type != None).all()]
    types     = [r[0] for r in db.query(distinct(PropertyModel.property_type)).filter(PropertyModel.property_type != None).all()]

    return {
        "offering_types": sorted(offerings),
        "property_types": sorted(types),
    }

# ── GET /options/all ───────────────────────────────────────────────────────────
import json

@router.get("/options/all")
def get_options():
    df = get_df()
    amenities_set = set()
    for val in df["amenities"].dropna():
        try:
            if isinstance(val, list):
                items = val
            elif isinstance(val, str) and val.strip().startswith("["):
                items = json.loads(val.replace("'", '"'))
            else:
                items = [v.strip() for v in str(val).split(",") if v.strip()]
            for item in items:
                cleaned = str(item).strip().strip("'\"")
                if cleaned:
                    amenities_set.add(cleaned)
        except Exception:
            pass

    def sorted_unique(col):
        return sorted(df[col].dropna().unique().tolist())

    return {
        "city":          sorted_unique("city"),
        "area":          sorted_unique("area"),
        "compound":      sorted_unique("compound"),
        "property_type": sorted_unique("property_type"),
        "offering_type": sorted_unique("offering_type"),
        "amenities":     sorted(amenities_set),
    }

# ── GET /{property_id} ─────────────────────────────────────────────────────────
@router.get("/{property_id}")
def get_property_details(
    property_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user)
):
    df = get_df()
    row = df[df["property_id"] == property_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Property not found")
    property_obj = row.iloc[0]
    return {
        k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
        for k, v in property_obj.to_dict().items()
    }

# ── ADMIN CRUD ─────────────────────────────────────────────────────────────────
from pydantic import BaseModel
from typing import Optional as Opt
from app.models.property import Property as PropertyModel

class PropertyCreate(BaseModel):
    title:          Opt[str]   = None
    description:    Opt[str]   = None
    property_type:  Opt[str]   = None
    offering_type:  Opt[str]   = None
    price:          Opt[float] = None
    price_per_sqm:  Opt[float] = None
    bedrooms:       Opt[int]   = None
    bathrooms:      Opt[int]   = None
    area_sqm:       Opt[float] = None
    city:           Opt[str]   = None
    area:           Opt[str]   = None
    compound:       Opt[str]   = None
    location_full:  Opt[str]   = None
    latitude:       Opt[float] = None
    longitude:      Opt[float] = None
    all_image_url:  Opt[str]   = None
    amenities:      Opt[str]   = None


def require_admin(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.post("/admin/refresh-cache")
def refresh_cache_endpoint(secret: str = None):
    """
    ✅ بينادى من ML pipeline (run_all.py) بعد ما الموديلات تتدرب
    عشان الـ FastAPI process يعمل reload للـ cache من DB/Redis مباشرة —
    من غير restart للسيرفر.
    """
    import os as _os
    expected = _os.getenv("CACHE_REFRESH_SECRET", "propertyhub-internal")
    if secret != expected:
        raise HTTPException(status_code=403, detail="Invalid secret")
    _refresh_cache()
    df = get_df()
    return {"message": "Cache refreshed", "properties_count": len(df)}


def _refresh_cache():
    """يعمل reload كامل للـ cache من DB مباشرة"""
    cache = get_cache()
    if cache.redis:
        try:
            cache.redis.delete("properties")
        except Exception:
            pass
    try:
        cache.properties = cache._get_df("properties", "SELECT * FROM properties_clean")
    except Exception:
        _db = SessionLocal()
        try:
            cache.properties = pd.read_sql("SELECT * FROM properties", _db.bind)
        finally:
            _db.close()
    print(f"✅ Cache refreshed — {len(cache.properties)} properties")


def _compute_derived_fields(prop_data: dict) -> dict:
    """
    يحسب الكولمنز المشتقة اللي مش جايه من الفورم:
    - price_per_sqm  = price / area_sqm
    - location_full  = "area, compound, city" (حسب المتاح)
    """
    price    = prop_data.get("price")
    area_sqm = prop_data.get("area_sqm")
    if price and area_sqm and area_sqm > 0:
        prop_data["price_per_sqm"] = round(price / area_sqm, 2)

    parts = [
        prop_data.get("area"),
        prop_data.get("compound"),
        prop_data.get("city"),
    ]
    parts = [p.strip() for p in parts if p and str(p).strip()]
    if parts:
        prop_data["location_full"] = ", ".join(parts)

    return prop_data


@router.post("/")
def add_property(
    data: PropertyCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    from datetime import datetime

    prop_data = data.dict(exclude_none=True)
    prop_data['updated_at'] = datetime.now()

    # ✅ احسب الكولمنز الناقصة (price_per_sqm, location_full)
    prop_data = _compute_derived_fields(prop_data)

    # ✅ احسب property_id يدوياً لو العمود مش AUTO_INCREMENT في الـ DB
    if "property_id" not in prop_data or not prop_data.get("property_id"):
        from sqlalchemy import func
        max_id = db.query(func.max(PropertyModel.property_id)).scalar()
        prop_data["property_id"] = (max_id or 0) + 1

    prop = PropertyModel(**prop_data)
    db.add(prop)
    db.commit()

    _refresh_cache()

    return {
        "message": "Property added",
        "property_id": prop_data["property_id"],
        "price_per_sqm": prop_data.get("price_per_sqm"),
        "location_full": prop_data.get("location_full"),
    }


@router.put("/{property_id}")
def update_property(
    property_id: int,
    data: PropertyCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    from datetime import datetime

    prop = db.query(PropertyModel).filter(PropertyModel.property_id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    update_data = data.dict(exclude_none=True)

    # ✅ خد القيم الحالية + الجديدة عشان نحسب price_per_sqm و location_full بشكل صحيح
    merged = {
        "price":    update_data.get("price",    prop.price),
        "area_sqm": update_data.get("area_sqm", prop.area_sqm),
        "area":     update_data.get("area",     prop.area),
        "compound": update_data.get("compound", prop.compound),
        "city":     update_data.get("city",     prop.city),
    }
    derived = _compute_derived_fields(dict(merged))
    update_data["price_per_sqm"] = derived.get("price_per_sqm", prop.price_per_sqm)
    update_data["location_full"] = derived.get("location_full", prop.location_full)

    for k, v in update_data.items():
        setattr(prop, k, v)

    prop.updated_at = datetime.now()

    db.commit()
    # ⚠️ متعملش db.refresh(prop) هنا — نفس مشكلة properties_clean

    _refresh_cache()
    return {"message": "Property updated", "property_id": property_id}


@router.delete("/{property_id}")
def delete_property(
    property_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    prop = db.query(PropertyModel).filter(PropertyModel.property_id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    db.delete(prop)
    db.commit()
    _refresh_cache()
    return {"message": "Property deleted"}