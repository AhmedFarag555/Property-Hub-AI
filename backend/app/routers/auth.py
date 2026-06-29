from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import random

from app.database.database import SessionLocal
from app.models.user import User
from app.models.otp_code import OTPCode
from app.schemas.user import UserCreate, UserLogin
from app.core.security import hash_password, verify_password, create_access_token
from app.core.email_service import send_otp_email

router = APIRouter(prefix="/auth", tags=["Auth"])


# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── helper: generate + store OTP ────────────────────────────────────────────
def _create_otp(db: Session, email: str, purpose: str) -> str:
    code = f"{random.randint(0, 999999):06d}"

    # امسح أي كود قديم لنفس الإيميل والـ purpose
    db.query(OTPCode).filter(
        OTPCode.email == email,
        OTPCode.purpose == purpose,
        OTPCode.used == False
    ).delete()

    otp = OTPCode(
        email=email,
        code=code,
        purpose=purpose,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(minutes=10),
        used=False
    )
    db.add(otp)
    db.commit()
    return code


def _verify_otp(db: Session, email: str, code: str, purpose: str) -> bool:
    otp = db.query(OTPCode).filter(
        OTPCode.email == email,
        OTPCode.code == code,
        OTPCode.purpose == purpose,
        OTPCode.used == False
    ).order_by(OTPCode.id.desc()).first()

    if not otp:
        return False
    if otp.expires_at < datetime.utcnow():
        return False

    otp.used = True
    db.commit()
    return True


# -------- REGISTER --------
@router.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):

    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")

    raw_password = user.password.strip()

    new_user = User(
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        phone=user.phone,
        password=hash_password(raw_password),
        is_verified=False
    )

    db.add(new_user)
    db.commit()

    # ✅ ابعت كود تأكيد على الإيميل
    code = _create_otp(db, user.email, "verify")
    send_otp_email(user.email, code, purpose="verify")

    return {
        "message": "User created successfully. Verification code sent to your email.",
        "email": user.email
    }


# -------- VERIFY EMAIL --------
from pydantic import BaseModel, EmailStr


class VerifyRequest(BaseModel):
    email: EmailStr
    code: str


@router.post("/verify-email")
def verify_email(data: VerifyRequest, db: Session = Depends(get_db)):
    ok = _verify_otp(db, data.email, data.code.strip(), "verify")
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_verified = True
    db.commit()

    return {"message": "Email verified successfully"}


# -------- RESEND VERIFICATION CODE --------
class ResendRequest(BaseModel):
    email: EmailStr


@router.post("/resend-code")
def resend_code(data: ResendRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")

    if user.is_verified:
        raise HTTPException(status_code=400, detail="Email already verified")

    code = _create_otp(db, data.email, "verify")
    sent = send_otp_email(data.email, code, purpose="verify")
    if not sent:
        raise HTTPException(status_code=500, detail="Failed to send email")

    return {"message": "Verification code resent"}


# -------- FORGOT PASSWORD: STEP 1 — send code --------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr


@router.post("/forgot-password")
def forgot_password(data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        # لا تكشف هل الإيميل موجود ولا لأ (أمان) — رجّع رسالة نفس الشكل
        return {"message": "If this email exists, a reset code has been sent."}

    code = _create_otp(db, data.email, "reset")
    send_otp_email(data.email, code, purpose="reset")

    return {"message": "If this email exists, a reset code has been sent."}


# -------- FORGOT PASSWORD: STEP 2 — verify code + set new password --------
class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str


@router.post("/reset-password")
def reset_password(data: ResetPasswordRequest, db: Session = Depends(get_db)):
    if len(data.new_password.strip()) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    ok = _verify_otp(db, data.email, data.code.strip(), "reset")
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password = hash_password(data.new_password.strip())
    db.commit()

    return {"message": "Password reset successfully. You can now log in."}


# -------- LOGIN --------
from fastapi.security import OAuth2PasswordRequestForm

@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    db_user = db.query(User).filter(User.email == form_data.username).first()

    if not db_user or not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # ✅ لازم الإيميل يكون متأكد (إلا الأدمن — مستثنى)
    if not db_user.is_admin and not db_user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please check your inbox for the verification code."
        )

    token = create_access_token({"user_id": db_user.user_id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "is_admin": bool(db_user.is_admin),
        "user_id": db_user.user_id,
        "first_name": db_user.first_name,
        "last_name": db_user.last_name
    }


from app.core.deps import get_current_user

@router.get("/me")
def get_me(current_user=Depends(get_current_user)):

    return {
        "user_id":    current_user.user_id,
        "email":      current_user.email,
        "first_name": current_user.first_name,
        "last_name":  current_user.last_name,
        "phone":      current_user.phone or "",
        "is_admin":   bool(current_user.is_admin),
        "is_verified": bool(current_user.is_verified)
    }


# -------- UPDATE PROFILE --------
from typing import Optional

class UserUpdate(BaseModel):
    first_name:   Optional[str] = None
    last_name:    Optional[str] = None
    phone:        Optional[str] = None
    email:        Optional[str] = None
    new_password: Optional[str] = None
    current_password: str


@router.put("/me")
def update_me(
    data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(data.current_password, user.password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    if data.first_name is not None: user.first_name = data.first_name.strip()
    if data.last_name  is not None: user.last_name  = data.last_name.strip()
    if data.phone      is not None: user.phone      = data.phone.strip()

    if data.email is not None and data.email.strip() != user.email:
        exists = db.query(User).filter(User.email == data.email.strip()).first()
        if exists:
            raise HTTPException(status_code=400, detail="Email already in use")
        user.email = data.email.strip()

    if data.new_password:
        if len(data.new_password.strip()) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        user.password = hash_password(data.new_password.strip())

    db.commit()
    db.refresh(user)

    return {
        "message":    "Profile updated successfully",
        "user_id":    user.user_id,
        "email":      user.email,
        "first_name": user.first_name,
        "last_name":  user.last_name,
        "phone":      user.phone or "",
    }