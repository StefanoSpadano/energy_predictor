import sqlite3
import pandas as pd
import numpy as np

def clean_data(df):
    df["temperature_measured"] = pd.to_numeric(df["temperature_measured"], errors="coerce")
    df["consumes"] = pd.to_numeric(df["consumes"], errors="coerce")
    df = df.dropna(subset=["consumes", "temperature_measured"])
    df = df[(df["temperature_measured"]>=-50.0) & (df["temperature_measured"] < 60)]
    df = df.reset_index(drop=True)
    return df

def run_pipeline(db_path="energy_data.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM energy_consumption", conn)
    df_clean = clean_data(df)
    df_clean.to_sql("clean_consumption", conn, if_exists = "replace", index = False)
    conn.close()

if __name__ == "__main__":
    run_pipeline()
    print("Database successfully cleansed")

