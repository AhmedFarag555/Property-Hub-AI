import pandas as pd
from datetime import datetime
from db import engine
from sqlalchemy import text
import redis
import json
from data_cache import cache

ACTION_WEIGHTS = {
    "view": 1,
    "click": 2,
    "favorite": 5,
    "contact": 10
}


class BehaviorService:

    def track(self, user_id, property_id, action):

        # ---------------- get property ----------------
        prop_df = pd.read_sql(
            text("SELECT * FROM properties_clean WHERE property_id = :property_id"),
            engine,
            params={"property_id": property_id}
        )
        
        VALID_ACTIONS = {"view", "click", "favorite", "contact"}

        if action not in VALID_ACTIONS:
            return {"error": "invalid action"}

        if prop_df.empty:
            return {"error": "property not found"}

        prop = prop_df.iloc[0]
        
        # =================================================
        # REALTIME SESSION ID
        # =================================================

        session_id = f"{user_id}_session"

        # ---------------- 1. SAVE INTERACTION ----------------
        pd.DataFrame([{
            "user_id": user_id,
            "property_id": property_id,
            "action": action,
            "created_at": datetime.now()
        }]).to_sql(
            "interactions",
            engine,
            if_exists="append",
            index=False
        )
        cache.refresh()
        
        # =================================================
        # REALTIME SESSION CACHE
        # =================================================

        r = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )

        session_key = f"session:{session_id}"

        existing_session = r.get(session_key)

        if existing_session:
            recent = json.loads(existing_session)
        else:
            recent = []

        recent.append({
            "property_id": int(property_id),
            "action": action,
            "timestamp": datetime.now().isoformat()
        })

        # keep only last 20 actions
        recent = recent[-20:]

        r.set(
            session_key,
            json.dumps(recent),
            ex=3600
        )

        # ---------------- 2. AUTO SAVE USER BEHAVIOR ----------------
        existing = pd.read_sql(
            text("""
                SELECT * 
                FROM user_behavior 
                WHERE user_id = :user_id 
                AND property_id = :property_id
            """),
            engine,
            params={
                "user_id": user_id,
                "property_id": property_id
            }
        )

        weight = ACTION_WEIGHTS.get(action, 1)

        if existing.empty:
            # INSERT
            pd.DataFrame([{
                "user_id": user_id,
                "property_id": property_id,
                "price": prop["price"],
                "bedrooms": prop["bedrooms"],
                "bathrooms": prop["bathrooms"],
                "area_sqm": prop["area_sqm"],
                "city": prop["city"],
                "compound": prop["compound"],
                "area": prop["area"],
                "property_type": prop["property_type"],
                "offering_type": prop["offering_type"],
                "interest_score": weight,
                "last_seen": datetime.now()
            }]).to_sql("user_behavior", engine, if_exists="append", index=False)

        else:
            # UPDATE
            new_score = existing["interest_score"].iloc[0] + weight

            with engine.begin() as conn:

                conn.execute(
                    text("""
                    UPDATE user_behavior
                    SET interest_score=:score,
                        last_seen=:last_seen
                    WHERE user_id=:user_id
                    AND property_id=:property_id
                    """),
                    {
                        "score": new_score,
                        "last_seen": datetime.now(),
                        "user_id": user_id,
                        "property_id": property_id
                    }
                )

        return {
            "status": "success",
            "message": "interaction tracked + behavior saved"
        }