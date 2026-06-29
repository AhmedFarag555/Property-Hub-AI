from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# لو عندك .env هنستخدمه بعدين
DATABASE_URL = "mysql+pymysql://root:1234@localhost/real_estate_db"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()