from fastapi import FastAPI
from sqlalchemy import text
from app.database.database import Base, engine
from app.routers import auth, properties
from app.routers.ml_price import router as ml_router
from app.routers.analytics import router as analytics_router
from app.routers.profile import router as profile_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

Base.metadata.create_all(bind=engine)

# ✅ UNIQUE KEY على interactions
with engine.connect() as _conn:
    try:
        _conn.execute(text("""
            ALTER TABLE interactions
            ADD UNIQUE KEY uq_user_prop_action (user_id, property_id, action)
        """))
        _conn.commit()
    except Exception:
        pass

# ✅ CORS — مش ممكن تجمع allow_origins=["*"] مع allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(properties.router)
app.include_router(ml_router)
app.include_router(analytics_router)
app.include_router(profile_router)

@app.get("/")
def home():
    return {"message": "API Running 🚀"}