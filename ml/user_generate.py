
import pandas as pd
import random
from passlib.context import CryptContext
from db import engine


# =========================
# PASSWORD HASHING
# =========================
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password):
    return pwd_context.hash(password)


# =========================
# RANDOM PHONE
# =========================
def random_phone():
    return f"+20{random.randint(1000000000, 1999999999)}"


# =========================
# GENERATE USERS
# =========================
def generate_users(n_users=200):

    users = []

    for i in range(2, n_users + 1):

        first_name = f"first_{i}"
        last_name = f"last_{i}"

        users.append({

            "user_id": i,

            "first_name": first_name,

            "last_name": last_name,

            "email": f"user{i}@test.com",

            "phone": random_phone(),

            # 🔥 IMPORTANT
            "password": hash_password("123456"),

            "preferred_lang": random.choice(["en", "ar"]),

            # 🔥 ADMIN RANDOM
            "is_admin": False

        })

    return pd.DataFrame(users)


# =========================
# INSERT
# =========================
if __name__ == "__main__":

    df = generate_users()

    df.to_sql(
        "users",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=500,
        method="multi"
    )

    print("Users inserted successfully ✔")

