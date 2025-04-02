import pandas as pd
import json
import base64
from datetime import datetime, timedelta
from IPython.display import display, HTML
import markdown
from markdown.extensions.tables import TableExtension
import re  # Added for regex search in messages

def format_duration(seconds):
    """
    Function to format duration from seconds to '[n]d, [n]h, [n]m, [n]s' format.
    """
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    duration_parts = []
    if days > 0:
        duration_parts.append(f"{days}d")
    if hours > 0:
        duration_parts.append(f"{hours}h")
    if minutes > 0:
        duration_parts.append(f"{minutes}m")
    if seconds > 0:
        duration_parts.append(f"{seconds}s")

    return ", ".join(duration_parts)

def svg_to_base64(svg_path):
    """
    Convert an SVG file to a base64 string.
    Used in main HTML to display avatars.
    """
    with open(svg_path, "rb") as svg_file:
        encoded_string = base64.b64encode(svg_file.read()).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded_string}"

# Step 1: Unpack the conversation
def unpack_conversation(df, conv_id):
    """
    Function to unpack the json formatted conversation turns in the
    input conv_id.
    Returns a dataframe of the conversation.
    """
    row = df[df['conv_id'] == conv_id]
    if row.empty:
        raise ValueError(f"No conversation found with conv_id: {conv_id}")

    conversation_data = row.iloc[0]['conversation']

    # Check if the data is a string or list and load accordingly
    if isinstance(conversation_data, str):
        conversation_data = json.loads(conversation_data)

    # Normalize the data into a dataframe if it's a list
    if isinstance(conversation_data, list):
        return pd.json_normalize(conversation_data)
    else:
        raise ValueError(f"Unexpected data format in conversation for conv_id: {conv_id}")

# Step 2: Format conversation metadata
def format_conversation_metadata(df, conv_id, source, search_phrase=None, search_count=0, search_turns=[]):
    """
    Function to format conversation metadata.

    Creates the info bubble for the conversation in the
    printed/saved output from main function.

    Output depends on the conv_id's source: different sources have
    different metadata.

    Currently: conditional formatting for sg and wc. More will be added if/when
    data is obtained from other source datasets.
    """
    row = df[df['conv_id'] == conv_id].iloc[0]

    # Common metadata
    common_metadata = f"""
    <p><b>Conversation ID:</b> {conv_id}</p>
    <p><b>User ID:</b> {row['user_id']} ({row['user_freq']} conversations)</p>
    <p><b>Model:</b> {row['model']}</p>
    """

    # Add search phrase metadata if applicable
    if search_phrase and search_count > 0:
        turn_numbers = ", ".join(map(str, [t + 1 for t in search_turns]))  # +1 for human-readable turns
        common_metadata += f"""
        <p><b>Number of messages with "{search_phrase}":</b> {search_count} (Turns {turn_numbers})</p>
        """

    if source == 'wc':
        # Duration calculation
        duration = pd.to_datetime(row['time_last']) - pd.to_datetime(row['time_first'])
        duration_str = format_duration(duration.total_seconds())

        # Turn information with styling
        turns_info = f"{row['turns']} (<span style='color:#C678DD;'>{row['code_turns']} code</span>, <span style='color:red;'>{row['toxic_turns']} toxic</span>, <span style='color:orange;'>{row['redacted_turns']} redacted</span>)"

        # Languages
        languages_str = row['language'] if row['n_languages'] == 1 else f"{row['n_languages']} ({row['language']})"

        # Final HTML with styling
        return f"""
        <div style="background-color: #1e1e1e; color: #ececec; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: sans-serif;">
            <h1 style="font-size: 1.5em; margin-bottom: 10px;">Conversation Details</h1>
            {common_metadata}
            <p><b>Turns:</b> {turns_info}</p>
            <p><b>Languages:</b> {languages_str}</p>
            <p><b>Start:</b> {row['time_first']}</p>
            <p><b>End:</b> {row['time_last']}</p>
            <p><b>Duration:</b> {duration_str}</p>
        </div>
        """
    elif source == 'sg':
        turns_info = f"{row['turns']} (<span style='color:#C678DD;'>{row['code_turns']} code</span>)"
        languages_str = row['language'] if row['n_languages'] == 1 else f"{row['n_languages']} ({row['language']})"

        return f"""
        <div style="background-color: #1e1e1e; color: #ececec; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: sans-serif;">
            <h1 style="font-size: 1.5em; margin-bottom: 10px;">Conversation Details</h1>
            {common_metadata}
            <p><b>Turns:</b> {turns_info}</p>
            <p><b>Languages:</b> {languages_str}</p>
            <p><b>Views:</b> {row['views']}</p>
        </div>
        """
    else:
        turns_info = f"{row['turns']} (<span style='color:#C678DD;'>{row['code_turns']} code</span>)"
        languages_str = row['language'] if row['n_languages'] == 1 else f"{row['n_languages']} ({row['language']})"

        return f"""
        <div style="background-color: #1e1e1e; color: #ececec; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: sans-serif;">
            <h1 style="font-size: 1.5em; margin-bottom: 10px;">Conversation Details</h1>
            {common_metadata}
            <p><b>Turns:</b> {turns_info}</p>
            <p><b>Languages:</b> {languages_str}</p>
        </div>
        """

def generate_chat_bubbles(conversation_df, source, search_phrase=None):
    """
    Function to generate the chat bubbles for the conversation with html styling.

    'source' determines output details. 'wc' conversations have
    turn-level tags for
    - time since last message
    - if message is toxic
    - if message is redacted
    """
    chat_html = []

    # Convert the avatars to base64
    assistant_avatar_base64 = svg_to_base64("/content/drive/MyDrive/Lab rotation 1/avatars/avatar-chatgpt.svg")
    user_avatar_base64 = svg_to_base64("/content/drive/MyDrive/Lab rotation 1/avatars/avatar-human.svg")

    initial_timestamp = None
    last_timestamp = None

    turn_number = 1  # Initialize turn number

    for idx, row in conversation_df.iterrows():
        role = row['role']
        message = row['message']
        language = row.get('language', None)
        timing_info = ""

        # Initialize flags
        flags = ""

        # Only for 'wc' source: Calculate timing info, toxic, and redacted flags
        if source == 'wc':
            if role == 'assistant' and 'timestamp' in row:
                current_timestamp = pd.to_datetime(row['timestamp'])
                if initial_timestamp is None:
                    initial_timestamp = current_timestamp
                else:
                    time_diff = (current_timestamp - last_timestamp).total_seconds()
                    timing_info = f"{format_duration(time_diff)} since last turn"
                last_timestamp = current_timestamp

            toxic = "<span style='color: red; font-weight: bold; float: right;'>(TOXIC)</span>" if row.get('toxic') else ""
            pii = "<span style='color: orange; font-weight: bold; float: right;'>(PII)</span>" if row.get('redacted') else ""
            flags = " ".join(filter(None, [pii, toxic]))

        # Language label for non-English languages
        language_label = ""
        if language and language.lower() != 'english':
            language_label = f"<div style='text-align: center; color: #FFD700; font-weight: bold;'>{language}</div>"

        # Check for search phrase in the message
        contains_search = False
        if search_phrase:
            if re.search(re.escape(search_phrase), message, re.IGNORECASE):
                contains_search = True
                # Highlight the search phrase in the message
                message = re.sub(
                    re.escape(search_phrase),
                    f"<strong style='color: red;'>{search_phrase}</strong>",
                    message,
                    flags=re.IGNORECASE
                )

        if role == 'assistant':
            # Convert Markdown to HTML with table and code support
            message_html = markdown.markdown(
                message,
                extensions=['fenced_code', TableExtension()]
            )

            bubble_class = "agent-turn"
            bubble_color = "#2A2A2A"
            avatar_html = f'<img src="{assistant_avatar_base64}" style="width: 40px; height: 40px; margin-right: 10px; border-radius: 50%;">'
            timing_html = f"<div style='font-size: 0.8em; color: #a0a0a0;'>{timing_info}</div>" if timing_info else ""
        else:
            message_html = message.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            bubble_class = "human-turn"
            bubble_color = "#474747"
            avatar_html = f'<img src="{user_avatar_base64}" style="width: 40px; height: 40px; margin-right: 10px; border-radius: 50%;">'
            timing_html = ""

        # Outline message in red if it contains the search phrase
        border_style = "border: 2px solid red;" if contains_search else ""

        bubble_html = f"""
        <div class="{bubble_class} clearfix" style="display: flex; align-items: flex-start; margin-bottom: 10px;">
            <div style="margin-right: 10px; font-weight: bold;">{turn_number}</div>
            {avatar_html}
            <div class="message" style="background-color: {bubble_color}; color: #ececec; padding: 15px; border-radius: 10px; flex-grow: 0; width: auto; max-width: 60%; font-family: sans-serif; line-height: 1.6; word-wrap: break-word; {border_style}">
                {timing_html}
                {language_label}
                <p style="margin: 0; padding: 0;">{flags}</p>
                <div style="margin: 0; padding: 5px 0;">{message_html}</div>
            </div>
        </div>
        """
        chat_html.append(bubble_html)
        turn_number += 1  # Increment turn number

    return "".join(chat_html)


def print_or_save_convo(df, conv_id, do_print=True, save=False, save_path='/content/drive/MyDrive/Lab rotation 1/html_convos/[conv_id].html', search_phrase=None):
    """
    Main function to print or save a conversation (conv id)
    from the input dataframe.

    HTML styling mimics styling in
    """
    # Select the row with the given conv_id
    row = df[df['conv_id'] == conv_id]
    if row.empty:
        display(HTML("<p>No conversation found.</p>"))
        return

    # Determine the source and unpack the conversation
    source = row.iloc[0]['source']
    conv_df = unpack_conversation(df, conv_id)

    # Initialize search-related variables
    search_count = 0
    search_turns = []

    if search_phrase:
        # Find messages containing the search phrase
        conv_df['contains_search'] = conv_df['message'].str.contains(re.escape(search_phrase), case=False, na=False)
        # Get the number of messages containing the search phrase
        search_count = conv_df['contains_search'].sum()
        # Get the turn numbers (zero-based index, so add 1 for human-readable)
        search_turns = conv_df[conv_df['contains_search']].index.tolist()

    # Format conversation metadata
    info_html = format_conversation_metadata(df, conv_id, source, search_phrase, search_count, search_turns)

    # Generate chat bubbles
    chat_html = generate_chat_bubbles(conv_df, source, search_phrase)

    # Combine metadata and chat bubbles into a full HTML document
    full_html = f"""
    <html>
    <head>
        <style>
            body {{
                background-color: #2A2A2A;
                color: #ececec;
                font-family: sans-serif;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            table, th, td {{
                border: 1px solid #7f8082;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                color: #ececec;
            }}
            th {{
                background-color: #3E3F4B;
            }}
            td {{
                background-color: #2C2C2F;
            }}
            pre {{
                background-color: #000000;
                padding: 10px;
                border-radius: 5px;
                color: #ececec;
                font-family: monospace;
                white-space: pre-wrap;
                margin: 0; /* Remove extra margins */
            }}
            code {{
                color: #ececec;
                font-family: monospace;
                padding: 2px 4px;
                border-radius: 4px;
                white-space: pre-wrap;
                margin: 0; /* Remove extra margins */
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        {info_html}
        <div class='conversation'>{chat_html}</div>
    </body>
    </html>
    """

    # Print the conversation if requested
    if do_print:
        display(HTML(full_html))

    # Save the conversation if requested
    if save:
        # Replace [conv_id] in save_path with the actual conversation ID
        save_path = save_path.replace('[conv_id]', conv_id)
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(full_html)
        print(f"Conversation saved as {save_path}")