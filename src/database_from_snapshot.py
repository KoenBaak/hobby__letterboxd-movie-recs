"""
This file was added in 2023 in order to create the database from a static
snapshot of csv files
"""
import os

import pandas as pd

from myapp import app, db


def create_from_snapshot():
    with app.app_context():
        for filename in os.listdir("database_snapshot"):
            df = pd.read_csv(f"database_snapshot/{filename}")
            df.to_sql(
                name=filename.replace(".csv", ""), con=db.engine, if_exists="replace"
            )
