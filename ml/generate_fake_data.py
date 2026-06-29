import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from db import engine


ACTION_PROB = {
    "view": 0.5,
    "click": 0.2,
    "favorite": 0.15,
    "contact": 0.15
}


def choose_action():
    return np.random.choice(
        list(ACTION_PROB.keys()),
        p=list(ACTION_PROB.values())
    )


def load_sources():

    users = pd.read_sql("SELECT user_id FROM users", engine)["user_id"].tolist()
    props = pd.read_sql("SELECT property_id FROM properties_clean", engine)["property_id"].tolist()

    return users, props


def generate_interactions(n_rows=20000):

    users, props = load_sources()

    rows = []

    for _ in range(n_rows):

        rows.append({
            "user_id": random.choice(users),
            "property_id": random.choice(props),
            "action": choose_action(),
            "created_at": datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    df = generate_interactions()

    # optional cleanup
    df = df.drop_duplicates(subset=["user_id", "property_id", "action"])

    df.to_sql(
        "interactions",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=1000,
        method="multi"
    )

    print("✔ interactions inserted with actions")