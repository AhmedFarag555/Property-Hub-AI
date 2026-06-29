"""
خدمة إرسال الإيميلات — تستخدم Gmail SMTP (مجاني).

كيفية الإعداد:
1. روح Google Account → Security → 2-Step Verification (لازم تكون مفعلة)
2. Security → App Passwords → اعمل App Password جديد (اسمه PropertyHUB مثلاً)
3. هيعطيك 16 رقم — حطهم في SMTP_PASSWORD تحت

لو معندكش Gmail، أي مزود SMTP تاني (Outlook, Yahoo...) هيشتغل بنفس الطريقة
بس غيّر SMTP_HOST و SMTP_PORT.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── الإعدادات — غيّرها بإيميلك ───────────────────────────────────────────────
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = "propertyhub878@gmail.com"        # ✅ إيميلك
SMTP_PASSWORD = "dtbb mmrl fysd cneo"          # ✅ الـ App Password (16 حرف بدون مسافات أو معاها)
FROM_NAME     = "PropertyHUB"


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """يبعت إيميل HTML. يرجع True لو نجح، False لو فشل (مع طبع الخطأ)."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{FROM_NAME} <{SMTP_USER}>"
        msg["To"]      = to_email

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_email, msg.as_string())

        print(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
        return False


def send_otp_email(to_email: str, code: str, purpose: str = "verify") -> bool:
    """
    purpose: 'verify' (تأكيد التسجيل) أو 'reset' (استعادة كلمة المرور)
    """
    if purpose == "reset":
        title = "إعادة تعيين كلمة المرور — PropertyHUB"
        intro = "طلبت إعادة تعيين كلمة المرور لحسابك. استخدم الكود ده لإكمال العملية:"
    else:
        title = "تأكيد البريد الإلكتروني — PropertyHUB"
        intro = "شكراً لتسجيلك في PropertyHUB! استخدم الكود ده لتأكيد بريدك الإلكتروني:"

    html = f"""
    <div style="font-family:'Segoe UI',Tahoma,sans-serif;max-width:480px;margin:0 auto;
                background:#fff;border-radius:16px;overflow:hidden;border:1px solid #eee">
      <div style="background:#D31148;padding:24px;text-align:center">
        <h1 style="color:#fff;margin:0;font-size:1.4rem;letter-spacing:1px">
          PROPERTY<span style="font-weight:300">HUB</span>
        </h1>
      </div>
      <div style="padding:32px 28px">
        <h2 style="margin:0 0 12px;font-size:1.1rem;color:#1a1a1a">{title}</h2>
        <p style="color:#666;font-size:14px;line-height:1.6;margin:0 0 24px">{intro}</p>
        <div style="background:#fdf2f5;border-radius:12px;padding:20px;text-align:center;margin-bottom:24px">
          <span style="font-size:32px;font-weight:700;letter-spacing:8px;color:#D31148">{code}</span>
        </div>
        <p style="color:#999;font-size:12px;line-height:1.6;margin:0">
          هذا الكود صالح لمدة 10 دقائق. إذا لم تطلب هذا الإجراء، يمكنك تجاهل هذه الرسالة بأمان.
        </p>
      </div>
      <div style="background:#f9f9f9;padding:16px;text-align:center;border-top:1px solid #eee">
        <p style="margin:0;font-size:11px;color:#aaa">© PropertyHUB — Egypt Real Estate Platform</p>
      </div>
    </div>
    """
    return send_email(to_email, title, html)