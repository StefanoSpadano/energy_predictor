from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sqlite3
import pandas as pd

def train_model(df=None, db_path="energy_data.db"):
    if df is None:
            conn = sqlite3.connect("energy_data.db")
            df = pd.read_sql_query("SELECT * FROM clean_consumption", conn)
            conn.close()
    X = df[["temperature_measured"]]
    y = df["consumes"]

    if len(df) < 5:
        raise ValueError("Dataset too small for training and testing split")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    error = mean_absolute_error(y_test, predictions)
    return error, model

if __name__ == "__main__":
    error, training_model = train_model()
    print(f"Model successfully trained returning {error:.2f} and {training_model}")
