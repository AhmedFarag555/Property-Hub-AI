import os
import sys
import pandas as pd
import time
import threading
import json

# ── path fix ──
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from db import engine


class DataCache:

    def __init__(self, refresh=False, ttl=300):
        self.ttl   = ttl
        self.redis = None
        self._pending_interactions = []   # in-memory real-time store
        self._interactions_db      = pd.DataFrame()

        # ── Redis optional ──
        try:
            import redis as redis_lib
            r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()
            self.redis = r
            print("✅ Redis connected")
        except Exception as e:
            print(f"⚠️  Redis not available ({e}) – running without cache")

        self.refresh()

        if refresh:
            threading.Thread(target=self.auto_refresh, daemon=True).start()

        print("✅ DataCache ready")

    # ---------------- INTERNAL LOAD ----------------
    def _get_df(self, key, query):
        # try Redis first
        if self.redis:
            try:
                cached = self.redis.get(key)
                if cached:
                    return pd.DataFrame(json.loads(cached))
            except Exception:
                pass

        # fallback: direct DB
        try:
            df = pd.read_sql(query, engine)
            if self.redis:
                try:
                    self.redis.setex(key, self.ttl, df.to_json(orient="records"))
                except Exception:
                    pass
            return df
        except Exception as e:
            print(f"❌ DB error on {key}: {e}")
            return pd.DataFrame()

    # ---------------- REFRESH ----------------
    def refresh(self):
        print("🔄 Refreshing cache...")
        self.properties   = self._get_df("properties",   "SELECT * FROM properties_clean")
        self._interactions_db = self._get_df("interactions", "SELECT * FROM interactions")
        self.behavior     = self._get_df("behavior",      "SELECT * FROM user_behavior")
        self.users        = self._get_df("users",         "SELECT * FROM users")

        # Merge DB interactions with in-memory pending ones
        self._rebuild_interactions()
        print("✅ Cache refreshed")

    def _rebuild_interactions(self):
        """Merge DB interactions + pending in-memory interactions."""
        base = self._interactions_db.copy() if hasattr(self, '_interactions_db') else pd.DataFrame()

        if self._pending_interactions:
            pending_df = pd.DataFrame(self._pending_interactions)
            base = pd.concat([base, pending_df], ignore_index=True)

        # Ensure required columns exist
        for col in ["user_id", "property_id", "action"]:
            if col not in base.columns:
                base[col] = pd.Series(dtype=object)

        self.interactions = base

    # ---------------- ADD INTERACTION (real-time) ----------------
    def add_interaction(self, user_id: int, property_id: int, action: str):
        """
        Add interaction immediately to in-memory store.
        No DB query needed — real-time update.
        """
        import datetime
        row = {
            "user_id":     int(user_id),
            "property_id": int(property_id),
            "action":      action,
            "created_at":  datetime.datetime.utcnow().isoformat()
        }
        self._pending_interactions.append(row)

        # Rebuild interactions immediately
        self._rebuild_interactions()
        print(f"⚡ Real-time interaction added: user={user_id} property={property_id} action={action} | total pending={len(self._pending_interactions)}")

    
    # ---------------- USER SESSION ----------------
    def get_user_session(self, user_id):
        if not self.redis:
            return []
        key = f"session:{user_id}"
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else []
        except Exception:
            return []

    # ---------------- AUTO REFRESH ----------------
    def auto_refresh(self, interval=300):
        while True:
            time.sleep(interval)
            self.refresh()

    # ---------------- HELPERS ----------------
    def get_properties(self):   return self.properties
    def get_interactions(self): return self.interactions
    def get_behavior(self):     return self.behavior
    def get_users(self):        return self.users


# ---------------- SINGLETON ----------------
cache = DataCache()