from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from app.database.database import Base


class UserBehavior(Base):
    __tablename__ = "user_behavior"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer)
    property_id   = Column(Integer)
    price         = Column(Float)
    bedrooms      = Column(Integer)
    bathrooms     = Column(Integer)
    area_sqm      = Column(Float)
    city          = Column(String(255))
    compound      = Column(String(255))
    area          = Column(String(255))
    property_type = Column(String(100))
    offering_type = Column(String(100))
    interest_score = Column(Float)
    last_seen     = Column(DateTime, default=datetime.utcnow)