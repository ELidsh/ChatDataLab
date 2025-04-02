import pandas as pd
import random
from typing import Optional, Union, List, Tuple


def filter_subset(df: pd.DataFrame,
                  return_all: bool = False,
                  conv_id_colname: str = 'conv_id',
                  **kwargs) -> Union[str, List[str], None]:
    """
    Return conversation ID(s) from the DataFrame that match the filters.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing conversation data (required positional argument).
    return_all : bool, default=False
        If True, returns all matching conversation IDs as a list.
        If False, returns a single random conversation ID.
    **kwargs : dict
        Keyword arguments for filtering. If a key matches a column name in df,
        filtering is applied based on the value type:
        - String columns:
          - Single value (e.g., source='wc')
          - List of values (e.g., source=['wc', 'other_source'])
        - Numerical columns:
          - Exact value (e.g., code_turns=0)
          - Range tuple:
            - (2, 10) means from 2 up to and including 10
            - (None, 10) means up to and including 10 (no lower bound)
            - (2, None) means 2 or more (no upper bound)

    Returns:
    --------
    str, List[str], or None:
        If return_all=False: A random conversation ID ('conv_id') from the filtered DataFrame.
        If return_all=True: A list of all matching conversation IDs.
        If no matching conversations are found, returns None.

    Example:
    --------
    # Get a random conversation with source 'wc', exactly 0 code turns,
    # and between 1-3 toxic turns
    filter_subset(df,
                  source='wc',
                  code_turns=0,
                  toxic_turns=(1, 3))

    # Get all conversations with at least 5 turns
    filter_subset(df, return_all=True, turns=(5, None))
    """

    # Helper function to parse range inputs
    def parse_range(range_input):
        """
        Parse range input and return (min, max) tuple.

        For numeric values:
        - int/float: exact value match
        - (2, 10): from 2 up to and including 10
        - (None, 10): up to and including 10 (no lower bound)
        - (2, None): 2 or more (no upper bound)
        """
        if range_input is None:
            return None, None

        # Handle exact value (int or float)
        if isinstance(range_input, (int, float)):
            return range_input, range_input

        # Handle tuple range
        if isinstance(range_input, tuple):
            if len(range_input) == 0:
                return None, None
            elif len(range_input) == 1:
                return range_input[0], None  # Only lower limit provided
            else:
                return range_input[0], range_input[1]  # Both limits provided

        # Default to exact match for anything else
        return range_input, range_input

    # Start with a copy of the DataFrame
    filtered_df = df.copy()

    # Apply filters for each keyword argument
    for key, value in kwargs.items():
        # Skip if the column doesn't exist
        if key not in df.columns:
            continue

        # Get the column data type
        dtype = df[key].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            # Numeric column handling
            min_val, max_val = parse_range(value)

            if min_val is not None and max_val is not None and min_val == max_val:
                # Exact value match
                filtered_df = filtered_df[filtered_df[key] == min_val]
            else:
                # Range filter
                if min_val is not None:
                    filtered_df = filtered_df[filtered_df[key] >= min_val]
                if max_val is not None:
                    filtered_df = filtered_df[filtered_df[key] <= max_val]
        else:
            # String/Object column handling
            if isinstance(value, list):
                # Filter with a list of values
                filtered_df = filtered_df[filtered_df[key].isin(value)]
            else:
                # Single value filter
                filtered_df = filtered_df[filtered_df[key] == value]

    # Check if we have any matches
    if filtered_df.empty:
        return None

    # Print the number of matching conversations
    print(f'{len(filtered_df)} conversations match filters')

    # Return based on return_all flag
    if return_all:
        return filtered_df[conv_id_colname].unique().tolist()
    else:
        return random.choice(filtered_df[conv_id_colname].unique())