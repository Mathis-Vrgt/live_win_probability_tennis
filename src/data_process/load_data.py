#python -m src.data_process.load_data

import pandas as pd
import os
from config.config_load import TOURNAMENT_INFO, POINTS_BY_ROUND


def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")

# test the function


if __name__ == "__main__":
    test_file_path = "/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2010s.csv"
    try:
        df = load_data(test_file_path)
        print("Data loaded successfully:")
        print(df.head())
    except Exception as e:
        print(e)

