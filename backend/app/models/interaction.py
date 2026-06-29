from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.database.database import Base


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer)        # ❌ بدون ForeignKey
    property_id = Column(Integer)    # ❌ بدون ForeignKey

    action = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)