from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from app.database.database import Base


class Property(Base):
    __tablename__ = "properties_clean"  # ✅ الجدول الصح
    __table_args__ = {"extend_existing": True}

    property_id = Column(Integer, primary_key=True, index=True)

    title = Column(String(255))
    description = Column(Text)

    property_type = Column(String(100))
    offering_type = Column(String(100))

    price = Column(Float)
    price_per_sqm = Column(Float)

    bedrooms = Column(Integer)
    bathrooms = Column(Integer)

    area_sqm = Column(Float)

    location_full = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)

    amenities = Column(Text)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    all_image_url = Column(Text)

    compound = Column(String(255))
    area = Column(String(255))
    city = Column(String(255))