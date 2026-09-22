import pandas as pd
import numpy as np 
import sqlite3

def gen_data(n_rows, start_date):
    "Function generating synthetic data with optional external input"
    time_stamps = pd.date_range(start=start_date, periods=n_rows, freq="h")
    temp_data = np.random.uniform(low=-10, high=39, size=n_rows)
    error = np.random.normal(0, 5, size=n_rows)


    optimal_distance = np.abs(temp_data - 18)
    base_load = 50
    scale_factor = 0.8
    consumes = base_load + (scale_factor * optimal_distance) + error


    df = pd.DataFrame({
    "timestamps": time_stamps, 
    "temperature_measured":temp_data, 
    "consumes": consumes
    })

    n_faults = max(1, int(n_rows*0.02))
    random_targeted_lines = np.random.randint(0, n_rows, size=n_faults)
    df.loc[random_targeted_lines, "consumes"] = np.nan
    df.loc[random_targeted_lines, "temperature_measured"] = 999.0
    return df  

def conn_sql(df):
    conn = sqlite3.connect("energy_data.db")
    df.to_sql("energy_consumption", conn, if_exists="replace", index = False)
    conn.close()

if __name__ == "__main__":
    print("Synthetic energetic data being generated")

    try:
        user_input_rows = input("How many lines do you want to generate? [Press Enter to default to 1000]: ")

        n_rows = int(user_input_rows) if user_input_rows.strip() != "" else 1000
    except ValueError:
        print("Input not valid; default value of 1000 selected.")
        n_rows = 1000

    start_date = input("Insert a start date (YYYY-MM-DD format) [Press Enter to default the value to 2025-01-01]: ")
    if start_date.strip() == "":
        start_date = "2025-01-01"
    
    print(f"{n_rows} hourly lectures being generated starting from {start_date}")
    my_df = gen_data(n_rows, start_date)
    conn_sql(my_df)
    print("Dataframe correctly created and saved to energy_data.db")
