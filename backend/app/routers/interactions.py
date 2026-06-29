from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

from app.database.database import SessionLocal
from app.schemas.interaction import InteractionCreate
from app.core.deps import get_current_user
from app.models.user import User

router = APIRouter(
    prefix="/interactions",
    tags=["Interactions"]
)


# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/")
def add_interaction(
    data: InteractionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    db.execute(
        text("""
        INSERT INTO interactions
        (
            user_id,
            property_id,
            action,
            created_at
        )
        VALUES
        (
            :user_id,
            :property_id,
            :action,
            :created_at
        )
        """),
        {
            "user_id": current_user.user_id,
            "property_id": data.property_id,
            "action": data.action,
            "created_at": datetime.now()
        }
    )

    db.commit()

    return {
        "message": "Interaction saved successfully",
        "user_id": current_user.user_id,
        "property_id": data.property_id
    }