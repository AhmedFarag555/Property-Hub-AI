from pydantic import BaseModel


class InteractionCreate(BaseModel):
    property_id: int
    action: str