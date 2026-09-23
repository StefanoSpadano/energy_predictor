# Energy Predictor

A Python application to generate, clean, and analyze synthetic energy consumption data using a machine learning model, with a strong focus on BDD unit testing and modularity.

## Functionalities
* **Synthetic data generation:** creates realistic energy consumption and temperature data mimicking real-world sensors;
* **ETL pipeline:** cleans missing values (NaNs), corrupted strings, and applies physical temperature boundaries;
* **Machine Learning prediction:** trains a Random Forest Regressor to predict energy consumption based on temperature;
* **Semantic testing:** comprehensive BDD-structured tests covering happy paths, edge cases, and deterministic behaviors.

### Pipeline steps
1. Data generation and SQLite injection (`database.py`);
2. Data extraction, cleaning, and re-injection (`pipeline.py`);
3. Model training and MAE evaluation (`model.py`).

## Structure of the project
Code is designed in modules:
* `database.py`: synthetic data generation and SQLite DB injection;
* `pipeline.py`: ETL logic, string parsing, and data cleansing;
* `model.py`: dataset splitting, Random Forest training, and error calculation;
* `tests/`: folder containing all the Pytest scripts (`test_database.py`, `test_pipeline.py`, `test_model.py`);
* `.gitignore`: handles the exclusion of virtual environments and local databases.

## System's requirements
In order to setup and run the project these are needed:
* Python 3 (es. `python3.10` or newer);
* The data manipulation libraries **Pandas** and **NumPy**;
* **Scikit-learn** for machine learning capabilities;
* **Pytest** for the testing suite.

## Setup & run
From the terminal navigate to the principal folder of the project and execute the following commands to create the environment and install dependencies: 
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn pytest
```

To run the BDD test suite:
```bash
python3 -m pytest
```

To execute the pipeline sequentially:
```bash
python3 database.py
python3 pipeline.py
python3 model.py
```

