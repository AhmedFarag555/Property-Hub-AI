from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.database.database import Base
from datetime import datetime


class OTPCode(Base):
    __tablename__ = "otp_codes"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    email       = Column(String(150), index=True)
    code        = Column(String(6))
    purpose     = Column(String(20))   # "verify" أو "reset"
    created_at  = Column(DateTime, default=datetime.utcnow)
    expires_at  = Column(DateTime)
    used        = Column(Boolean, default=False)