import os
import sys
import joblib
import numpy as np
from datetime import datetime

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from data_cache import cache


ACTION_WEIGHTS = {
    "view": 1,
    "click": 2,
    "favorite": 6,
    "contact": 12
}


class UserVectorBuilder:

    def __init__(self, data):

        self.df       = data["df"]
        self.features = data["features"]

        # property_id → positional index in features matrix
        self.id_to_index = {
            pid: idx
            for idx, pid in enumerate(self.df["property_id"])
        }

        # ── precompute property_id → positional index in cache.properties ──
        # (used for profile bias lookups)

    # ------------------------------------------------
    # SAFE POSITIONAL INDEX LOOKUP
    # ------------------------------------------------
    def _get_feature_rows(self, df_subset):
        """
        Given a subset of cache.properties, return the list of
        *positional* indices into self.features for rows that
        exist in the model's df.
        """
        ids = df_subset["property_id"].values
        return [self.id_to_index[pid] for pid in ids if pid in self.id_to_index]

    # ------------------------------------------------
    # BUILD USER VECTOR (HYBRID)
    # ------------------------------------------------
    def build_user_vector(self, user_id):

        interactions = cache.get_interactions()

        user_actions = interactions[
            interactions["user_id"] == user_id
        ]

        recent_session = cache.get_user_session(user_id)

        if user_actions.empty and not recent_session:
            return None

        weighted_vectors = []
        total_weight     = 0

        SESSION_WEIGHTS = {
            "view": 1,
            "click": 3,
            "favorite": 6,
            "contact": 12
        }

        # =================================================
        # HISTORY VECTOR
        # =================================================
        for _, row in user_actions.iterrows():

            property_id = row["property_id"]
            action      = row["action"]
            weight      = ACTION_WEIGHTS.get(action, 1)

            if property_id not in self.id_to_index:
                continue

            idx = self.id_to_index[property_id]
            weighted_vectors.append(self.features[idx] * weight)
            total_weight += weight

        # =================================================
        # SESSION VECTOR (WITH TIME DECAY)
        # =================================================
        session_vectors = []
        session_total   = 0

        for item in recent_session:

            property_id = item["property_id"]
            action      = item["action"]
            weight      = SESSION_WEIGHTS.get(action, 1)

            if "timestamp" in item:
                try:
                    ts        = datetime.fromisoformat(item["timestamp"])
                    hours_old = (datetime.now() - ts).total_seconds() / 3600
                    decay     = np.exp(-hours_old / 24)
                except Exception:
                    decay = 1.0
            else:
                decay = 1.0

            weight = weight * decay * 2.0  # session boost

            if property_id not in self.id_to_index:
                continue

            idx = self.id_to_index[property_id]
            session_vectors.append(self.features[idx] * weight)
            session_total += weight

        # =================================================
        # SAFETY CHECK
        # =================================================
        if total_weight == 0 and session_total == 0:
            return None

        # =================================================
        # BUILD HISTORY VECTOR
        # =================================================
        if total_weight > 0:
            behavior_vector = sum(weighted_vectors) / total_weight
        else:
            behavior_vector = np.zeros((1, self.features.shape[1]))

        # =================================================
        # MERGE SESSION + HISTORY
        # =================================================
        if session_total > 0:
            session_vector  = sum(session_vectors) / session_total
            behavior_vector = 0.6 * behavior_vector + 0.4 * session_vector

        # =================================================
        # PROFILE BIAS
        # =================================================
        properties = cache.properties

        user_props = properties[
            properties["property_id"].isin(user_actions["property_id"])
        ]

        profile_bias = np.zeros((1, self.features.shape[1]))

        if not user_props.empty:

            # ── area preference ──
            top_area = user_props["area"].mode()
            if not top_area.empty:
                area_subset  = properties[properties["area"] == top_area.iloc[0]]
                area_pos_ids = self._get_feature_rows(area_subset)
                if area_pos_ids:
                    area_vec      = self.features[area_pos_ids].mean(axis=0)
                    profile_bias += area_vec * 0.12

            # ── type preference ──
            top_type = user_props["property_type"].mode()
            if not top_type.empty:
                type_subset  = properties[properties["property_type"] == top_type.iloc[0]]
                type_pos_ids = self._get_feature_rows(type_subset)
                if type_pos_ids:
                    type_vec      = self.features[type_pos_ids].mean(axis=0)
                    profile_bias += type_vec * 0.10

        # =================================================
        # FINAL VECTOR
        # =================================================
        user_vector = np.asarray(behavior_vector + profile_bias).ravel()
        norm        = np.linalg.norm(user_vector)
        user_vector = user_vector / (norm + 1e-9)

        # =================================================
        # META FEATURES
        # =================================================
        top_city    = None
        top_type_v  = None
        recent_city = None

        if not user_props.empty:
            city_mode = user_props["city"].mode()
            if not city_mode.empty:
                top_city = city_mode.iloc[0]

            type_mode = user_props["property_type"].mode()
            if not type_mode.empty:
                top_type_v = type_mode.iloc[0]

        avg_price = float(user_props["price"].mean())    if not user_props.empty else None
        avg_lat   = float(user_props["latitude"].mean()) if not user_props.empty else None
        avg_lon   = float(user_props["longitude"].mean()) if not user_props.empty else None

        # recent city from session
        recent_property_ids = [x["property_id"] for x in recent_session]
        recent_props        = self.df[self.df["property_id"].isin(recent_property_ids)]

        if not recent_props.empty:
            rc_mode = recent_props["city"].mode()
            if not rc_mode.empty:
                recent_city = rc_mode.iloc[0]

        if recent_city is None:
            recent_city = top_city

        return {
            "vector":      user_vector.astype(np.float32),
            "avg_price":   avg_price,
            "fav_city":    top_city,
            "recent_city": recent_city,
            "fav_type":    top_type_v,
            "avg_lat":     avg_lat,
            "avg_lon":     avg_lon,
            "user_props":  user_props,
        }