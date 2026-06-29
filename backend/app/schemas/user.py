from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    phone: str
    password: str = Field(min_length=4, max_length=72)


class UserLogin(BaseModel):
    email: EmailStr
    password: str