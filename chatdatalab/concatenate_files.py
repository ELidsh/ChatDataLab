import pandas as pd
import glob
import os

def concatenate_files(directory, file_type = 'json'):
    """
    Reads all .json files in the specified directory, and concatenates them into a single DataFrame.

    Parameters:
    directory (str): The directory containing the .json files.

    Returns:
    pd.DataFrame: A DataFrame containing the concatenated data from all .json files.
    """
    # Use glob to find all .json files in the directory
    json_files = glob.glob(os.path.join(directory, f"*.{file_type}"))

    # Create a list to hold dataframes
    dataframes = []

    # Iterate through the json files and read each one into a dataframe
    for file in json_files:
        df = pd.read_json(file, orient='records', lines=True)
        dataframes.append(df)

    # Concatenate all the dataframes into a single dataframe
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    return concatenated_df