from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# connection
engine = create_engine(
    "mysql+mysqlconnector://root:1234@localhost/real_estate_db",
    pool_pre_ping=True
)

# load tables
sale_df = pd.read_sql("SELECT * FROM properties_sale", engine)
rent_df = pd.read_sql("SELECT * FROM properties_rent", engine)

print("Sale shape:", sale_df.shape)
print("Rent shape:", rent_df.shape)

# merge
df = pd.concat([sale_df, rent_df], ignore_index=True)

print("Merged shape:", df.shape)



try:

    with engine.begin() as conn:

        df.to_sql(
            name="properties",
            con=conn,
            if_exists="replace",
            index=False,
            chunksize=1000,
            method="multi"
        )

    print("All data merged successfully!")

except SQLAlchemyError as e:
    print("SQL Error:")
    print(e)

except Exception as e:
    print("General Error:")
    print(e)