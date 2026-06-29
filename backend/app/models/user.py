from sqlalchemy import Column, Integer, String, Boolean
from app.database.database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)

    first_name = Column(String(100))
    last_name = Column(String(100))

    email = Column(String(150), unique=True, index=True)

    phone = Column(String(20))

    password = Column(String(255))


    is_admin = Column(Boolean, default=False)

    is_verified = Column(Boolean, default=False)