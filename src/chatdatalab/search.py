import pandas as pd
import random
import re
import warnings
from typing import Union, List, Tuple


def search_text_matches(df: pd.DataFrame,
                        text: str,
                        case_sensitive: bool = True,
                        from_start: bool = False,
                        return_all: bool = False,
                        **kwargs) -> Union[List[str], Tuple[str, List[int]], None]:
    """
    Search for text matches in a DataFrame's 'message' column and apply additional filters.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing conversation data with at least 'message', 'conv_id', and 'turn_num' columns.
    text : str
        The text to search for in the 'message' column.
    case_sensitive : bool, default=True
        Whether the text search should be case sensitive.
    from_start : bool, default=False
        If True, only search for text matches at the beginning of messages.
        If False, search for matches anywhere in the message.
    return_all : bool, default=False
        If True, returns a list of all unique 'conv_id' values from matching rows.
        If False, returns a tuple with (random conv_id, list of turn_num values with matches in that conv_id).
    **kwargs : dict
        Additional keyword arguments for filtering. If a key matches a column name in df,
        filtering is applied using the same logic as in filter_subset:
        - String columns (like 'role'): single value or list of values
          e.g., role='user' to filter for user messages only
        - Numerical columns: exact value or range tuple
        A warning will be issued for kwargs that don't match column names.

    Returns:
    --------
    - If return_all=True: List[str] of unique conv_ids matching the search
    - If return_all=False: Tuple[str, List[int]] containing (random conv_id, list of turn_nums with matches)
    - If no matches found: None

    Example:
    --------
    # Find a random conversation with "Python" in messages from the assistant
    search_text_matches(df, "Python", role="assistant")

    # Find all conversations where messages start with "Hello"
    search_text_matches(df, "Hello", from_start=True, return_all=True)

    # Find all conversations with "help" in messages and at least 5 turns
    search_text_matches(df, "help", return_all=True, turns=(5, None))
    """

    # Helper function to parse range inputs for numerical filtering
    def parse_range(range_input):
        """Parse range input and return (min, max) tuple."""
        if range_input is None:
            return None, None

        if isinstance(range_input, (int, float)):
            return range_input, range_input

        if isinstance(range_input, tuple):
            if len(range_input) == 0:
                return None, None
            elif len(range_input) == 1:
                return range_input[0], None
            else:
                return range_input[0], range_input[1]

        return range_input, range_input

    # Verify required columns exist
    required_columns = ['message', 'conv_id', 'turn_num']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame is missing one or more required columns: {required_columns}")

    # Start with a copy of the DataFrame
    filtered_df = df.copy()

    # Apply text search to message column
    if from_start:
        # Search for text only at the beginning of messages
        if case_sensitive:
            # Use regex with ^ to match the start of the string
            pattern = f"^{re.escape(text)}"
            filtered_df = filtered_df[filtered_df['message'].str.match(pattern, na=False)]
        else:
            # Case-insensitive match from the start
            pattern = f"^{re.escape(text)}"
            filtered_df = filtered_df[filtered_df['message'].str.match(pattern, case=False, na=False)]
    else:
        # Search for text anywhere in messages
        if case_sensitive:
            filtered_df = filtered_df[filtered_df['message'].str.contains(text, na=False)]
        else:
            filtered_df = filtered_df[filtered_df['message'].str.contains(text, case=False, na=False)]

    # Apply additional filters from kwargs
    for key, value in kwargs.items():
        # Warn if the column doesn't exist
        if key not in df.columns:
            warnings.warn(f"Column '{key}' not found in DataFrame. This filter will be ignored.")
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

    # Print the number of matching rows and conversations
    unique_convs = filtered_df['conv_id'].unique()
    print(f'Found {len(filtered_df)} matching messages in {len(unique_convs)} conversations')

    # Return based on return_all flag
    if return_all:
        return unique_convs.tolist()
    else:
        # Select a random conversation
        random_conv = random.choice(unique_convs)

        # Get all turn_num values with matches in this conversation
        matching_turns = filtered_df[filtered_df['conv_id'] == random_conv]['turn_num'].tolist()

        return (random_conv, matching_turns)