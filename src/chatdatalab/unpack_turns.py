
def unpack_conversations(df, conv_column = 'conversation'):
    # Step 1: Create a subset of df with only the 'conversation' column
    conversation_df = df[[conv_column]].copy()

    # Step 2: Unpack each conversation dictionary to one row per dict
    # Assuming each entry in the 'conversation' column is a list of dictionaries
    unpacked_df = conversation_df.explode(conv_column)

    # Convert the dictionaries in the 'conversation' column into separate columns
    unpacked_df = unpacked_df.dropna().reset_index(drop=True)
    unpacked_df = pd.json_normalize(unpacked_df[conv_column])

    return unpacked_df