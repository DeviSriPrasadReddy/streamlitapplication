# need to check this is for fixed conversations and visual multiline things 
import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context, no_update, clientside_callback, dash_table
import dash_bootstrap_components as dbc
import json
import logging
import uuid # <-- ADDED for unique message IDs

# --- MODIFICATION: Import dash.flask to access request headers ---
import flask

# --- MODIFICATION: Import your updated functions ---
# These functions are assumed to be in a file named 'genie_room.py'
# You will need to create this file if you haven't already.
# For this example, we'll create placeholder functions if they don't exist.
try:
    from genie_room import genie_query, record_feedback
except ImportError:
    print("WARNING: 'genie_room.py' not found. Using placeholder functions.")
    
    def genie_query(user_input, conversation_id, user_token=None):
        """Placeholder function for genie_query."""
        print(f"Placeholder Query: {user_input}, ConvID: {conversation_id}, Token: {user_token}")
        if "table" in user_input.lower() or "sales" in user_input.lower():
            # Return a sample DataFrame
            df = pd.DataFrame({
                "day": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07"]),
                "sales": [150, 200, 250, 220, 300, 280, 400],
                "profit": [30, 45, 50, 40, 60, 55, 70],
                "region": ["North", "South", "East", "West", "North", "South", "East"]
            })
            return "conv-123", "msg-456", df.to_dict('records'), "SELECT * FROM sales_data"
        else:
            # Return a sample string response
            return "conv-123", "msg-456", "This is a text response to your query.", None

    def record_feedback(conv_id, msg_id, feedback_type, user_token=None):
        """Placeholder function for record_feedback."""
        print(f"Placeholder Feedback: ConvID: {conv_id}, MsgID: {msg_id}, Type: {feedback_type}, Token: {user_token}")
        pass
# --- End of Placeholder ---

import pandas as pd
import os
from dotenv import load_dotenv
import sqlparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import numpy as np # <-- ADDED for numeric checks

load_dotenv()
# --- ADDED: Set up logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- MODIFIED: Helper function to get user info ---
def get_user_info_from_header():
    """Reads the user token and user initial from the request headers."""
    token = None
    initial = "Y" # Default initial
    try:
        # --- !!!  ACTION REQUIRED  !!! ---
        # You may need to change this header name.
        # Common names: 'X-Databricks-User-Token', 'X-Forwarded-Access-Token'
        token_header_name = 'X-Forwarded-Access-Token'
        token = flask.request.headers.get(token_header_name)

        # --- NEW: Get User Email/Name ---
        # Common names: 'X-Forwarded-User', 'X-Databricks-User'
        user_header_name = 'X-Forwarded-User'
        user_email = flask.request.headers.get(user_header_name)
        
        if user_email and len(user_email) > 0:
            initial = user_email[0].upper()
            
    except Exception as e:
        # This will fail if not in a request context (e.g., on startup)
        logger.debug(f"Not in request context or header not found: {e}")
        pass
    
    if not token:
        logger.warning(f"Could not find user token in request headers. Will use fallback.")
        
    return token, initial
# --- END MODIFICATION ---


# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],title ="GENIE AI"
)
server = app.server # Expose server for Gunicorn

# ... (DEFAULT_WELCOME_TITLE, DEFAULT_SUGGESTIONS are all unchanged) ...
DEFAULT_WELCOME_TITLE = "Supply Chain Optimization"
DEFAULT_WELCOME_DESCRIPTION = "Analyze your Supply Chain Performance leveraging AI/BI Dashboard. Deep dive into your data and metrics."
DEFAULT_SUGGESTIONS = [
    "What tables are there and how are they connected? Give me a short summary.",
    "Which distribution center has the highest chance of being a bottleneck?",
    "Show me a table of sales and profit by day.",
    "What was the demand for our products by week in 2024?"
]

# Define the layout
app.layout = html.Div([
    # Top navigation bar
    html.Div([
        # Left component containing both nav-left and sidebar
        html.Div([
            # Nav left
            html.Div([
                html.Button([
                    html.Img(src="assets/menu_icon.svg", className="menu-icon")
                ], id="sidebar-toggle", className="nav-button"),
                html.Button([
                    html.Img(src="assets/plus_icon.svg", className="new-chat-icon")
                ], id="new-chat-button", className="nav-button",disabled=False),
                html.Button([
                    html.Img(src="assets/plus_icon.svg", className="new-chat-icon"),
                    html.Div("New chat", className="new-chat-text")
                ], id="sidebar-new-chat-button", className="new-chat-button",disabled=False)
            ], id="nav-left", className="nav-left"),
            
            # Sidebar
            html.Div([
                html.Div([
                    html.Div("Your conversations with Genie", className="sidebar-header-text"),
                ], className="sidebar-header"),
                html.Div([], className="chat-list", id="chat-list")
            ], id="sidebar", className="sidebar")
        ], id="left-component", className="left-component"),
        
        html.Div([
            html.Div("Genie Space", id="logo-container", className="logo-container")
        ], className="nav-center"),
        html.Div([
            # --- MODIFIED: Added ID to user-avatar div ---
            html.Div("Y", id="user-avatar", className="user-avatar"),
            html.A(
                html.Button(
                    "Logout",
                    id="logout-button",
                    className="logout-button"
                ),
                href=f"https://{os.getenv('DATABRICKS_HOST')}/login.html",
                className="logout-link"
            )
        ], className="nav-right")
    ], className="top-nav"),
    
    # Main content area
    html.Div([
        html.Div([
            # Chat content
            html.Div([
                # Welcome container
                html.Div([
                    html.Div([html.Div([
                        html.Div(className="genie-logo")
                    ], className="genie-logo-container")],
                    className="genie-logo-container-header"),
                    
                    # Add settings button with tooltip
                    html.Div([
                        html.Div(id="welcome-title", className="welcome-message", children=DEFAULT_WELCOME_TITLE),
                        html.Button([
                            html.Img(src="assets/settings_icon.svg", className="settings-icon"),
                            html.Div("Customize welcome message", className="button-tooltip")
                        ],
                        id="edit-welcome-button",
                        className="edit-welcome-button",
                        title="Customize welcome message")
                    ], className="welcome-title-container"),
                    
                    html.Div(id="welcome-description", 
                             className="welcome-message-description",
                             children=DEFAULT_WELCOME_DESCRIPTION),
                    
                    # Add modal for editing welcome text
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Customize Welcome Message")),
                        dbc.ModalBody([
                            html.Div([
                                html.Label("Welcome Title", className="modal-label"),
                                dbc.Input(
                                    id="welcome-title-input",
                                    type="text",
                                    placeholder="Enter a title for your welcome message",
                                    className="modal-input"
                                ),
                                html.Small(
                                    "This title appears at the top of your welcome screen",
                                    className="text-muted d-block mt-1"
                                )
                            ], className="modal-input-group"),
                            html.Div([
                                html.Label("Welcome Description", className="modal-label"),
                                dbc.Textarea(
                                    id="welcome-description-input",
                                    placeholder="Enter a description that helps users understand the purpose of your application",
                                    className="modal-input",
                                    style={"height": "80px"}
                                ),
                                html.Small(
                                    "This description appears below the title and helps guide your users",
                                    className="text-muted d-block mt-1"
                                )
                            ], className="modal-input-group"),
                            html.Div([
                                html.Label("Suggestion Questions", className="modal-label"),
                                html.Small(
                                    "Customize the four suggestion questions that appear on the welcome screen",
                                    className="text-muted d-block mb-3"
                                ),
                                dbc.Input(
                                    id="suggestion-1-input",
                                    type="text",
                                    placeholder="First suggestion question",
                                    className="modal-input mb-2"
                                ),
                                dbc.Input(
                                    id="suggestion-2-input",
                                    type="text",
                                    placeholder="Second suggestion question",
                                    className="modal-input mb-2"
                                ),
                                dbc.Input(
                                    id="suggestion-3-input",
                                    type="text",
                                    placeholder="Third suggestion question",
                                    className="modal-input mb-2"
                                ),
                                dbc.Input(
                                    id="suggestion-4-input",
                                    type="text",
                                    placeholder="Fourth suggestion question",
                                    className="modal-input"
                                )
                            ], className="modal-input-group")
                        ]),
                        dbc.ModalFooter([
                            dbc.Button(
                                "Cancel",
                                id="close-modal",
                                className="modal-button",
                                color="light"
                            ),
                            dbc.Button(
                                "Save Changes",
                                id="save-welcome-text",
                                className="modal-button-primary",
                                color="primary"
                            )
                        ])
                    ], id="edit-welcome-modal", is_open=False, size="lg", backdrop="static"),
                    
                    # Suggestion buttons with IDs
                    html.Div([
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[0], 
                                     className="suggestion-text", id="suggestion-1-text")
                        ], id="suggestion-1", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[1],
                                     className="suggestion-text", id="suggestion-2-text")
                        ], id="suggestion-2", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[2],
                                     className="suggestion-text", id="suggestion-3-text")
                        ], id="suggestion-3", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[3],
                                     className="suggestion-text", id="suggestion-4-text")
                        ], id="suggestion-4", className="suggestion-button")
                    ], className="suggestion-buttons")
                ], id="welcome-container", className="welcome-container visible"),
                
                # Chat messages
                html.Div([], id="chat-messages", className="chat-messages"),
            ], id="chat-content", className="chat-content"),
            
            # Input area
            html.Div([
                html.Div([
                    dcc.Input(
                        id="chat-input-fixed",
                        placeholder="Ask your question...",
                        className="chat-input",
                        type="text",
                        disabled=False
                    ),
                    html.Div([
                        html.Button(
                            id="send-button-fixed", 
                            className="input-button send-button",
                            disabled=False
                        )
                    ], className="input-buttons-right"),
                    html.Div("You can only submit one query at a time", 
                           id="query-tooltip", 
                           className="query-tooltip hidden")
                ], id="fixed-input-container", className="fixed-input-container"),
                html.Div("Always review the accuracy of responses.", className="disclaimer-fixed")
            ], id="fixed-input-wrapper", className="fixed-input-wrapper"),
        ], id="chat-container", className="chat-container"),
    ], id="main-content", className="main-content"),
    
    html.Div(id='dummy-output'),
    dcc.Store(id="chat-trigger", data={"trigger": False, "message": "", "conversation_id": None}),
    dcc.Store(id="chat-history-store", data=[]),
    dcc.Store(id="query-running-store", data=False),
    # Modified session-store to hold both UI session index and backend conversation_id
    dcc.Store(id="session-store", data={"current_session": None, "conversation_id": None}),
    # --- NEW: Added store to trigger user initial load ---
    dcc.Store(id="initial-load-trigger", data=0)
])

def format_sql_query(sql_query):
    """Format SQL query using sqlparse library"""
    return sqlparse.format(
        sql_query, keyword_case='upper', reindent=True, indent_width=2
    )

def call_llm_for_insights(df, prompt=None):
    """
    Call an LLM to generate insights from a DataFrame.
    NOTE: This uses WorkspaceClient() which will use the environment variables.
    This call will likely run as the Service Principal, NOT the user.
    """
    if prompt is None:
        prompt = (
            "You are a professional data analyst. Given the following table data, "
            "provide deep, actionable analysis for 1. Key insights and trends 2. Notable patterns and" 
            " anomalies 3. Business implications."
            "Be thorough, professional, and concise.\n\n"
        )
    csv_data = df.to_csv(index=False)
    full_prompt = f"{prompt}Table data:\n{csv_data}"
    try:
        client = WorkspaceClient() # Uses SP env vars
        
        # --- THIS PAYLOAD IS FOR CHAT MODELS ---
        # If your endpoint is NOT a chat model, you may need to change this
        # to: request={"prompt": full_prompt}
        response = client.serving_endpoints.query(
            os.getenv("SERVING_ENDPOINT_NAME"),
            messages=[ChatMessage(content=full_prompt, role=ChatMessageRole.USER)],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return f"Error generating insights: {str(e)}"
    

# --- NEW FUNCTION: To get follow-up suggestions ---
def get_followup_questions(user_query: str, bot_response: str) -> dict:
    """
    Calls a serving endpoint to get follow-up question suggestions.
    """
    # --- !!! ACTION REQUIRED !!! ---
    # Add a new variable to your .env file:
    # SUGGESTION_ENDPOINT_NAME=your-suggestion-model-endpoint-name
    SUGGESTION_ENDPOINT_NAME = os.environ.get("SUGGESTION_ENDPOINT_NAME")
    if not SUGGESTION_ENDPOINT_NAME:
        logger.warning("SUGGESTION_ENDPOINT_NAME not set. Skipping follow-up questions.")
        return {}

    # The prompt asks for JSON output for easy parsing.
    prompt = f"""
    Given a user's question and a chatbot's answer, generate one "better" version of the user's question and two relevant follow-up questions.
    Return ONLY a single valid JSON object with the keys "better_prompt", "followup1", and "followup2".

    User Question: "{user_query}"
    Chatbot Answer: "{bot_response[:1000]}"

    JSON:
    """

    try:
        client = WorkspaceClient() # Uses SP env vars
        
        # --- !!! ACTION REQUIRED !!! ---
        # This payload assumes a model that accepts a 'prompt'.
        # If your model expects 'messages', change this to:
        # request={"messages": [{"role": "user", "content": prompt}]}
        payload = {
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.5
        }

        response = client.serving_endpoints.query(
            SUGGESTION_ENDPOINT_NAME,
            request=payload
        )

        # Parse the response. This is highly dependent on your model.
        # This assumes the model returns: {"predictions": ["{\"better_prompt\": ...}"]}
        if "predictions" in response and response.predictions:
            # Clean up potential markdown/fencing
            json_str = response.predictions[0].strip().replace("```json", "").replace("```", "")
            data = json.loads(json_str)
            return {
                "better_prompt": data.get("better_prompt"),
                "followup1": data.get("followup1"),
                "followup2": data.get("followup2"),
            }
        else:
            logger.warning(f"Unexpected response from suggestion endpoint: {response}")
            return {}

    except Exception as e:
        logger.error(f"Error getting follow-up questions: {e}")
        return {}
# --- END NEW FUNCTION ---


# --- MODIFIED FUNCTION: To get visual spec ---
def get_visual_spec(df: pd.DataFrame, user_query: str) -> dict: # <-- ADDED user_query
    """
    Calls a serving endpoint to generate a Plotly JSON chart specification
    from a DataFrame, guided by the user's query.
    """
    # --- !!! ACTION REQUIRED !!! ---
    # Add a new variable to your .env file:
    # VISUAL_ENDPOINT_NAME=your-visual-model-endpoint-name
    # For now, we'll re-use the main serving endpoint name
    VISUAL_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME")
    if not VISUAL_ENDPOINT_NAME:
        logger.warning("VISUAL_ENDPOINT_NAME not set. Skipping visualization.")
        return "Error: Visualization endpoint is not configured."

    # --- MODIFIED PROMPT ---
    prompt = f"""
    You are a data visualization expert. Your task is to generate a Plotly JSON specification
    for a chart that best visualizes the provided data, based on the user's request.

    **User's Request:** "{user_query}"

    **Instructions:**
    1.  Analyze the user's request. If they *specifically* ask for a chart type
        (e.g., "pie chart", "bar chart", "line graph"), you MUST generate that chart type.
    2.  If no specific chart type is requested, choose the *most appropriate* chart
        type based on the data's schema and content (e.g., bar for categorical,
        line for time-series).
    
    **Multi-Metric Plotting:**
    3.  If the user's request mentions multiple metrics (e.g., "plot sales AND profit over time"),
        you MUST plot them on the *same* chart. Use the 'color' aesthetic to differentiate them.
    4.  For wide-form data (e.g., columns 'date', 'metric1', 'metric2'), you must generate a
        spec that plots both 'metric1' and 'metric2' against 'date'. Add two separate traces
        in the 'data' array if necessary, one for each metric, and use 'name' to label them.

    **Data Schema (dtypes):**
    {df.dtypes.to_string()}

    **Data (first 5 rows):**
    {df.head().to_string()}
    
    CRITICAL RULES: 
    1: Your response MUST be a single valid JSON object
    2: Do NOT include any text, preamble, explanation, or conversational phrases.
    3. DO NOT include Markdown fences or code blocks 
    4. The JSON must be perfectly valid and parasable 
    5. The reponse must start with '{{' and end with '}}'. 
        
    Respond ONLY with a single valid JSON object in the format:
    {{"data": [...], "layout": {{...}}}}
    """
    # --- END MODIFIED PROMPT ---

    try:
        client = WorkspaceClient() # Uses SP env vars
        
        response = client.serving_endpoints.query(
            VISUAL_ENDPOINT_NAME,
            messages=[ChatMessage(content=prompt, role=ChatMessageRole.USER)] # <-- MODIFIED
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error getting visual spec: {e}")
        return f"Error getting visual spec: {str(e)}"
# --- END MODIFIED FUNCTION ---

# --- NEW HELPER FUNCTIONS ---
def get_last_dataframe(chat_history, session_data):
    """Finds the most recent DataFrame JSON for the current session."""
    try:
        current_session_index = session_data.get("current_session", 0)
        if current_session_index is None: current_session_index = 0
        
        if chat_history and current_session_index < len(chat_history):
            dataframes = chat_history[current_session_index].get('dataframes')
            if dataframes:
                # Get the most recent table ID from this session
                last_df_id = max(dataframes.keys()) 
                return dataframes[last_df_id]
    except Exception as e:
        logger.error(f"Error getting last dataframe: {e}")
    return None

def is_visual_refinement_query(user_query):
    """Checks if a query is likely a follow-up about visualization."""
    query = user_query.lower()
    refinement_keywords = [
        'pie chart', 'bar chart', 'line chart', 'scatter plot',
        'as a ', 'visualize this', 'plot this', 'show me a '
    ]
    return any(kw in query for kw in refinement_keywords)
# --- END NEW HELPER FUNCTIONS ---


# --- MODIFIED CALLBACK 1: Handle inputs and show thinking indicator ---
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-input-fixed", "value", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input("suggestion-1", "n_clicks"),
     Input("suggestion-2", "n_clicks"),
     Input("suggestion-3", "n_clicks"),
     Input("suggestion-4", "n_clicks"),
     Input("send-button-fixed", "n_clicks"),
     Input("chat-input-fixed", "n_submit"),
     Input({"type": "followup-suggestion", "index": ALL}, "n_clicks")],
    [State("suggestion-1-text", "children"),
     State("suggestion-2-text", "children"),
     State("suggestion-3-text", "children"),
     State("suggestion-4-text", "children"),
     State("chat-input-fixed", "value"),
     State("chat-messages", "children"),
     State("welcome-container", "className"),
     State("chat-list", "children"),
     State("chat-history-store", "data"),
     State("session-store", "data"),
     # --- MODIFICATION: Add new State for follow-up suggestions ---
     State({"type": "followup-suggestion", "index": ALL}, "id")
    ],
    prevent_initial_call=True
)
def handle_all_inputs(s1_clicks, s2_clicks, s3_clicks, s4_clicks, send_clicks, submit_clicks,
                      followup_clicks, # <--- New argument
                      s1_text, s2_text, s3_text, s4_text, input_value, current_messages,
                      welcome_class, current_chat_list, chat_history, session_data,
                      followup_ids): # <--- New argument
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * 8

    trigger_id_str = ctx.triggered[0]["prop_id"].split(".")[0]
    
    suggestion_map = {
        "suggestion-1": s1_text, "suggestion-2": s2_text,
        "suggestion-3": s3_text, "suggestion-4": s4_text
    }
    
    user_input = None

    if "followup-suggestion" in trigger_id_str:
        try:
            # User clicked one of the new follow-up buttons
            trigger_id_dict = json.loads(trigger_id_str)
            user_input = trigger_id_dict.get("text")
        except:
            pass # Ignore if parse fails
    elif trigger_id_str in suggestion_map:
        # User clicked a welcome suggestion
        user_input = suggestion_map[trigger_id_str]
    else:
        # User typed or clicked send
        user_input = input_value
    # --- END MODIFICATION ---

    if not user_input:
        return [no_update] * 8
    
    # Create user message
    user_message = html.Div([
        html.Div([html.Div("Y", className="user-avatar"), html.Span("You", className="model-name")], className="user-info"),
        html.Div(user_input, className="message-text")
    ], className="user-message message")
    updated_messages = (current_messages or []) + [user_message]
    
    # Add thinking indicator
    thinking_indicator = html.Div([
        html.Div([html.Span(className="spinner"), html.Span("Thinking...")], className="thinking-indicator")
    ], className="bot-message message")
    updated_messages.append(thinking_indicator)
    
    # --- MODIFIED: Handle session management ---
    current_session_index = session_data.get("current_session")
    current_conv_id = session_data.get("conversation_id")
    chat_history = chat_history or []

    if current_session_index is None: # "New Chat" was clicked
        current_session_index = 0 # New chat will be at index 0
        # Create the new session
        new_session = {
            "session_id": 0, # Placeholder, will be re-indexed
            "backend_conversation_id": None, # A new chat has no backend ID yet
            "queries": [user_input],
            "messages": updated_messages,
            "dataframes": {} # NEW: Add empty dataframes dict
        }
        chat_history.insert(0, new_session) # Insert at the front
        
        # Re-index all sessions
        for i, session in enumerate(chat_history):
            session["session_id"] = i
        
        current_conv_id = None
    
    else: # Update existing chat
        # The current_session_index from session_data IS the correct index
        if current_session_index < len(chat_history):
            chat_history[current_session_index]["messages"] = updated_messages
            chat_history[current_session_index]["queries"].append(user_input)
            current_conv_id = chat_history[current_session_index].get("backend_conversation_id")
        else:
            # This case shouldn't happen, but if it does, treat as new chat
            logger.warning(f"Session index {current_session_index} out of bounds. Creating new chat.")
            current_session_index = 0
            new_session = {
                "session_id": 0,
                "backend_conversation_id": None,
                "queries": [user_input],
                "messages": updated_messages,
                "dataframes": {}
            }
            chat_history.insert(0, new_session)
            for i, session in enumerate(chat_history): 
                session["session_id"] = i
            current_conv_id = None
    # --- END MODIFIED SESSION LOGIC ---

    # Update chat list UI
    updated_chat_list = []
    for i, session in enumerate(chat_history): # 'i' is now the correct index
        first_query = session["queries"][0] if session.get("queries") else "Empty Chat"
        is_active = (i == current_session_index) 
        updated_chat_list.append(
            html.Div(
                first_query,
                className=f"chat-item{' active' if is_active else ''}",
                id={"type": "chat-item", "index": i} # The ID index MUST match the loop index
            )
        )
    
    # Pass conversation_id to the trigger
    trigger_data = {
        "trigger": True, 
        "message": user_input,
        "conversation_id": current_conv_id # Pass the current backend ID
    }
    updated_session_data = {
        "current_session": current_session_index, 
        "conversation_id": current_conv_id
    }

    return (updated_messages, "", "welcome-container hidden",
            trigger_data, True,
            updated_chat_list, chat_history, updated_session_data)
# --- END MODIFIED CALLBACK 1 ---


# --- MODIFIED CALLBACK 2 ---
# Callback 2: Make API call and show response
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input("chat-trigger", "data")],
    [State("chat-messages", "children"),
     State("chat-history-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def get_model_response(trigger_data, current_messages, chat_history, session_data):
    if not trigger_data or not trigger_data.get("trigger"):
        return no_update, no_update, no_update, no_update, no_update
    
    user_input = trigger_data.get("message", "")
    conversation_id = trigger_data.get("conversation_id")
    if not user_input:
        return no_update, no_update, no_update, no_update, no_update
    
    # --- MODIFIED: Get user token from new helper ---
    user_token, _ = get_user_info_from_header()
    # --- END MODIFICATION ---

    current_session_index = session_data.get("current_session", 0)
    if current_session_index is None:
        current_session_index = 0

    try:
        # --- NEW LOGIC: Check for Visual Refinement Query ---
        last_df_json = get_last_dataframe(chat_history, session_data)
        
        if is_visual_refinement_query(user_input) and last_df_json:
            logger.info("Visual refinement query detected. Bypassing genie_query.")
            df = pd.read_json(last_df_json, orient='split')
            
            # 1. Get new visual spec
            visual_spec_raw = get_visual_spec(df, user_input)
            visual_spec_clean = visual_spec_raw.strip().replace("json","").replace("```","").replace("'",'"').replace("\\n","").replace("\r","")
            
            try:
                visual_spec = json.loads(visual_spec_clean)
                content = html.Div([
                    dcc.Markdown("Here's the refined visual:"),
                    dcc.Graph(
                        figure=visual_spec,
                        style={'width': '95%', 'height': '400px'},
                        config={'responsive': True}
                    )
                ])
            except Exception as e:
                logger.error(f"Failed to parse visual spec on refinement: {e}")
                content = dcc.Markdown(f"I tried to create that visual, but failed: {str(e)}\n\n`{visual_spec_clean}`")

            # 2. Create a new bot response for this visual
            conv_id = session_data.get("conversation_id", "session-no-id")
            msg_id = f"vis-refine-{uuid.uuid4()}" 

            bot_response = html.Div([
                html.Div([html.Div(className="model-avatar"), html.Span("Genie", className="model-name")], className="model-info"),
                html.Div([
                    content,
                    html.Div([
                        html.Div([
                            html.Button(id={"type": "thumbs-up", "index": msg_id, "conv_id": conv_id}, className="thumbs-up-button"),
                            html.Button(id={"type": "thumbs-down", "index": msg_id, "conv_id": conv_id}, className="thumbs-down-button")
                        ], className="message-actions")
                    ], className="message-footer")
                ], className="message-content")
            ], className="bot-message message")

            updated_messages = (current_messages or [])[:-1] + [bot_response]
            
            if chat_history and current_session_index < len(chat_history):
                chat_history[current_session_index]["messages"] = updated_messages
            
            new_session_data = {**session_data, "conversation_id": conv_id}
            
            return updated_messages, chat_history, {"trigger": False, "message": ""}, False, new_session_data
        
        # --- END OF NEW LOGIC ---

        # --- REGULAR FLOW: Call genie_query ---
        conv_id, msg_id, response, query_text = genie_query(
            user_input, 
            conversation_id, 
            user_token=user_token
        )
        
        if conv_id is None:
            error_msg = str(response)
            error_response = html.Div([
                html.Div([html.Div(className="model-avatar"), html.Span("Genie", className="model-name")], className="model-info"),
                html.Div([html.Div(error_msg, className="message-text")], className="message-content")
            ], className="bot-message message")
            
            new_session_data = {"current_session": None, "conversation_id": None}
            updated_messages = (current_messages or [])[:-1] + [error_response]
            return updated_messages, chat_history, {"trigger": False, "message": ""}, False, new_session_data

        new_session_data = {**session_data, "conversation_id": conv_id}
        
        response_text_for_context = "" # For follow-up prompt
        
        if isinstance(response, str):
            content = dcc.Markdown(response, className="message-text")
            response_text_for_context = response
        else:
            # --- IT'S A DATAFRAME: AUTOMATICALLY VISUALIZE ---
            df = pd.DataFrame(response)
            # --- NEW: Convert date columns ---
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        logger.info(f"Converted column '{col}' to datetime.")
                    except (ValueError, TypeError):
                        pass # Not a date string
            # --- END NEW ---

            df_id = f"table-{len(chat_history)}-{len(current_messages)}"
            
            # Store DF in history
            if not chat_history or current_session_index >= len(chat_history):
                 chat_history.insert(0, {"dataframes": {}, "queries": [], "messages": []}) # Ensure session exists
            
            chat_history[current_session_index].setdefault('dataframes', {})[df_id] = df.to_json(orient='split')

            # Create Table
            data_table = dash_table.DataTable(
                id=df_id, data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                export_format="csv", export_headers="display", page_size=10,
                style_table={'overflowX': 'auto', 'width': '95%'},
                style_cell={'textAlign': 'left', 'fontSize': '12px', 'padding': '4px 10px', 'fontFamily': 'sans-serif'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': '600'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                fill_width=False
            )
            
            # Create Query Section
            query_section = None
            if query_text is not None:
                formatted_sql = format_sql_query(query_text)
                query_index = f"{len(chat_history)}-{len(current_messages)}"
                query_section = html.Div([
                    html.Div([
                        html.Button([
                            html.Span("Show code", id={"type": "toggle-text", "index": query_index})
                        ], id={"type": "toggle-query", "index": query_index}, className="toggle-query-button", n_clicks=0)
                    ], className="toggle-query-container"),
                    html.Div(
                        html.Pre(html.Code(formatted_sql, className="sql-code")),
                        id={"type": "query-code", "index": query_index}, className="query-code-container hidden"
                    )
                ], id={"type": "query-section", "index": query_index}, className="query-section")
            
            # --- NEW: Generate Visual Automatically ---
            graph_id = {"type": "dynamic-graph", "index": df_id} # <-- NEW ID
            visual_spec_raw = get_visual_spec(df, user_input)
            visual_spec_clean = visual_spec_raw.strip().replace("json","").replace("```","").replace("'",'"').replace("\\n","").replace("\r","")
            visual_output = None
            try:
                visual_spec = json.loads(visual_spec_clean)
                visual_output = dcc.Graph(
                    id=graph_id, # <-- SET ID
                    figure=visual_spec,
                    style={'width': '95%', 'height': '400px', 'marginTop': '15px'},
                    config={'responsive': True}
                )
            except Exception as e:
                logger.error(f"Failed to parse visual spec on initial load: {e}")
                visual_output = dcc.Markdown(f"Error generating visual: {str(e)}\n\n`{visual_spec_clean}`", className="insight-content")
            # --- END NEW VISUAL LOGIC ---

            # --- NEW: Create Dynamic Filters ---
            filter_components = []
            filter_style = {"display": "inline-block", "marginRight": "15px", "minWidth": "200px"}
            
            # 1. Date filter
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if date_cols:
                date_col = date_cols[0] # Use first date column
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                filter_components.append(html.Div([
                    html.Label("Date Range:", style={"fontWeight": "bold", "display": "block", "marginBottom": "5px"}),
                    dcc.DatePickerRange(
                        id={"type": "date-picker", "index": df_id},
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=min_date,
                        end_date=max_date,
                        display_format='YYYY-MM-DD',
                        className="date-picker-genie"
                    )
                ], style=filter_style))

            # 2. Threshold filter
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                filter_components.append(html.Div([
                    html.Label("Threshold (>=):", style={"fontWeight": "bold", "display": "block", "marginBottom": "5px"}),
                    dcc.Dropdown(
                        id={"type": "threshold-column", "index": df_id},
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        placeholder="Select column...",
                        style={"marginBottom": "5px"}
                    ),
                    dbc.Input(
                        id={"type": "threshold-input", "index": df_id},
                        type="number",
                        placeholder="Enter value..."
                    )
                ], style=filter_style))
            
            filter_bar = html.Div(filter_components, className="filter-bar", style={"padding": "10px", "backgroundColor": "#f9f9f9", "borderRadius": "5px", "margin": "10px 0"})
            # --- END NEW FILTERS ---

            # Create Insight Button
            insight_button = html.Button(
                "Generate Insights",
                id={"type": "insight-button", "index": df_id},
                className="insight-button",
                style={"border": "none", "background": "#f0f0f0", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}
            )
            insight_output = dcc.Loading(
                id={"type": "insight-loading", "index": df_id},
                type="circle",
                children=html.Div(id={"type": "insight-output", "index": df_id})
            )
            
            # Button container (Removed visual button)
            button_container = html.Div([
                insight_button,
            ], style={"display": "flex", "marginTop": "10px"})
            
            # --- NEW: Selection Info Box ---
            selection_info = html.Div(
                id={"type": "selection-info", "index": df_id},
                children=html.P("Use the Box Select or Lasso Select tool on the chart to see details here.", style={"fontStyle": "italic", "color": "#888", "padding": "10px", "borderTop": "1px solid #eee", "marginTop": "10px"}),
                className="selection-info-box"
            )
            # --- END NEW ---
            
            # Assemble content
            content = html.Div([
                html.Div(data_table, style={'marginBottom': '20px', 'paddingRight': '5px'}),
                filter_bar, # <-- ADDED FILTERS
                visual_output, # <-- This is the dcc.Graph
                selection_info, # <-- ADDED SELECTION INFO BOX
                query_section if query_section else None,
                button_container, 
                insight_output,
                dcc.Store(id={"type": "query-store", "index": df_id}, data=user_input) # <-- NEW
            ])
            
            response_text_for_context = f"A table and visual were returned for the query: {query_text}"

        
        # --- Follow-up suggestions (runs for both text and df responses) ---
        
        # Define inline styles for the new suggestion buttons
        pill_style = {
            "backgroundColor": "#f0f0f0", "border": "1px solid #ddd",
            "borderRadius": "16px", "padding": "6px 12px", "margin": "4px",
            "fontSize": "13px", "cursor": "pointer", "display": "block",
            "textAlign": "left", "width": "fit-content", "maxWidth": "100%",
            "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
            "lineHeight": "1.4"
        }
        prefix_style = {"fontWeight": "600", "marginRight": "5px"}
        container_style = {"paddingTop": "10px", "marginTop": "10px", "borderTop": "1px solid #eee"}
        
        suggestion_div = html.Div(style=container_style)
        try:
            suggestions = get_followup_questions(user_input, response_text_for_context)
            suggestion_elements = []
            
            if suggestions.get("better_prompt"):
                suggestion_elements.append(
                    html.Button([
                        html.Span("💡 Better way to ask: ", style=prefix_style),
                        html.Span(suggestions["better_prompt"])
                    ], id={"type": "followup-suggestion", "index": 0, "text": suggestions["better_prompt"]},
                       style=pill_style)
                )
            if suggestions.get("followup1"):
                suggestion_elements.append(
                    html.Button([
                        html.Span("Relevant question: ", style=prefix_style),
                        html.Span(suggestions["followup1"])
                    ], id={"type": "followup-suggestion", "index": 1, "text": suggestions["followup1"]},
                       style=pill_style)
                )
            if suggestions.get("followup2"):
                suggestion_elements.append(
                    html.Button([
                        html.Span("Relevant question: ", style=prefix_style),
                        html.Span(suggestions["followup2"])
                    ], id={"type": "followup-suggestion", "index": 2, "text": suggestions["followup2"]},
                       style=pill_style)
                )
            
            if suggestion_elements:
                suggestion_div = html.Div(suggestion_elements, style=container_style)
            else:
                suggestion_div = html.Div() # Empty div, no border

        except Exception as e:
            logger.error(f"Failed to generate follow-up suggestions: {e}")
            suggestion_div = html.Div() # Empty div on failure
        # --- END MODIFICATION ---

        # Create bot response
        bot_response = html.Div([
            html.Div([html.Div(className="model-avatar"), html.Span("Genie", className="model-name")], className="model-info"),
            html.Div([
                content,
                html.Div([
                    html.Div([
                        html.Button(id={"type": "thumbs-up", "index": msg_id, "conv_id": conv_id}, className="thumbs-up-button"),
                        html.Button(id={"type": "thumbs-down", "index": msg_id, "conv_id": conv_id}, className="thumbs-down-button")
                    ], className="message-actions")
                ], className="message-footer"),
                
                # --- MODIFICATION: Add the new suggestion div ---
                suggestion_div
                # --- END MODIFICATION ---

            ], className="message-content")
        ], className="bot-message message")
        
        updated_messages = (current_messages or [])[:-1] + [bot_response]
        
        if chat_history and current_session_index < len(chat_history):
            chat_history[current_session_index]["messages"] = updated_messages
            chat_history[current_session_index]["backend_conversation_id"] = conv_id
        else:
             # This case should be handled by the logic above, but as a fallback
             chat_history.insert(0, {
                 "session_id": current_session_index,
                 "backend_conversation_id": conv_id,
                 "queries": [user_input],
                 "messages": updated_messages,
                 "dataframes": chat_history[current_session_index].get('dataframes', {}) # Preserve DFs if they exist
             })

        return updated_messages, chat_history, {"trigger": False, "message": ""}, False, new_session_data
        
    except Exception as e:
        logger.error(f"Error in get_model_response: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again later."
        error_response = html.Div([
            html.Div([html.Div(className="model-avatar"), html.Span("Genie", className="model-name")], className="model-info"),
            html.Div([html.Div(error_msg, className="message-text")], className="message-content")
        ], className="bot-message message")
        
        updated_messages = (current_messages or [])[:-1] + [error_response]
        if chat_history and current_session_index < len(chat_history):
            chat_history[current_session_index]["messages"] = updated_messages
        
        return updated_messages, chat_history, {"trigger": False, "message": ""}, False, no_update
# --- END MODIFIED CALLBACK 2 ---


# Callback 3: Toggle sidebar
@app.callback(
    [Output("sidebar", "className"),
     Output("new-chat-button", "style"),
     Output("sidebar-new-chat-button", "style"),
     Output("logo-container", "className"),
     Output("nav-left", "className"),
     Output("left-component", "className"),
     Output("main-content", "className")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className"),
     State("left-component", "className"),
     State("main-content", "className")]
)
def toggle_sidebar(n_clicks, s_class, l_class, m_class):
    if n_clicks:
        if "sidebar-open" in s_class:
            return "sidebar", {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", "main-content"
        else:
            return "sidebar sidebar-open", {"display": "none"}, {"display": "flex"}, "logo-container logo-container-open", "nav-left nav-left-open", "left-component left-component-open", "main-content main-content-shifted"
    return s_class, {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", l_class, m_class

# Callback 4: Chat item selection
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input({"type": "chat-item", "index": ALL}, "n_clicks")],
    [State("chat-history-store", "data"),
     State("chat-list", "children"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def show_chat_history(n_clicks, chat_history, current_chat_list, session_data):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks):
        return no_update, no_update, no_update, no_update
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    clicked_index = json.loads(triggered_id)["index"]
    
    if not chat_history or clicked_index >= len(chat_history):
        return no_update, no_update, no_update, no_update
    
    # --- MODIFIED: Ensure index is valid ---
    if clicked_index < 0 or clicked_index >= len(chat_history):
        logger.warning(f"Clicked index {clicked_index} out of bounds for chat history.")
        return no_update, no_update, no_update, no_update
    
    clicked_session_data = chat_history[clicked_index]
    new_conv_id = clicked_session_data.get("backend_conversation_id")
    new_session_data = {"current_session": clicked_index, "conversation_id": new_conv_id}
    
    updated_chat_list = []
    for i, item in enumerate(current_chat_list):
        new_class = "chat-item active" if i == clicked_index else "chat-item"
        updated_chat_list.append(
            html.Div(item["props"]["children"], className=new_class, id={"type": "chat-item", "index": i})
        )
    
    return (chat_history[clicked_index]["messages"], "welcome-container hidden", 
            updated_chat_list, new_session_data)
# --- END MODIFIED CALLBACK 4 ---

# Callback 5: Clientside scroll to bottom
app.clientside_callback(
    """
    function(children) {
        var chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    Input('chat-messages', 'children'),
    prevent_initial_call=True
)

# Callback 6: New chat button
@app.callback(
    [Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True)],
    [Input("new-chat-button", "n_clicks"),
     Input("sidebar-new-chat-button", "n_clicks")],
    [State("chat-history-store", "data"),
     State("chat-list", "children")],
    prevent_initial_call=True
)
def reset_to_welcome(n1, n2, chat_history_store, chat_list):
    new_session_data = {"current_session": None, "conversation_id": None}
    
    updated_chat_list = []
    if chat_list:
        # --- MODIFIED: Rebuild list, but remove 'active' class ---
        for i, item in enumerate(chat_list):
            updated_chat_list.append(
                html.Div(item["props"]["children"], className="chat-item", id={"type": "chat-item", "index": i})
            )
    else:
        updated_chat_list = no_update
    # --- END MODIFICATION ---
    
    return ("welcome-container visible", [], {"trigger": False, "message": ""}, 
            False, new_session_data, updated_chat_list)

# (Callback 7 removed, was redundant)
@app.callback(
    [Output("welcome-container", "className", allow_duplicate=True)],
    [Input("chat-messages", "children")],
    prevent_initial_call=True
)
def hide_welcome_on_chat(chat_messages):
    if chat_messages:
        return ["welcome-container hidden"]
    else:
        return ["welcome-container visible"]

# Callback 8: Disable input while query is running
@app.callback(
    [Output("chat-input-fixed", "disabled"),
     Output("send-button-fixed", "disabled"),
     Output("new-chat-button", "disabled"),
     Output("sidebar-new-chat-button", "disabled"),
     Output("query-tooltip", "className")],
    [Input("query-running-store", "data")],
    prevent_initial_call=True
)
def toggle_input_disabled(query_running):
    tooltip_class = "query-tooltip visible" if query_running else "query-tooltip hidden"
    return query_running, query_running, query_running, query_running, tooltip_class


# Callback 9: Handle feedback
@app.callback(
    [Output({"type": "thumbs-up", "index": MATCH, "conv_id": MATCH}, "className"),
     Output({"type": "thumbs-down", "index": MATCH, "conv_id": MATCH}, "className")],
    [Input({"type": "thumbs-up", "index": MATCH, "conv_id": MATCH}, "n_clicks"),
     Input({"type": "thumbs-down", "index": MATCH, "conv_id": MATCH}, "n_clicks")],
    prevent_initial_call=True
)
def handle_feedback(up_clicks, down_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    
    trigger_id = ctx.triggered_id
    button_type = trigger_id["type"]
    msg_id = trigger_id["index"]
    conv_id = trigger_id["conv_id"]
    
    # --- MODIFIED: Get user token from new helper ---
    user_token, _ = get_user_info_from_header()
    # --- END MODIFICATION ---
    
    if button_type == "thumbs-up":
        logger.info(f"Recording POSITIVE feedback for conv_id: {conv_id}, msg_id: {msg_id}")
        record_feedback(conv_id, msg_id, "POSITIVE", user_token=user_token)
        return "thumbs-up-button active", "thumbs-down-button"
        
    elif button_type == "thumbs-down":
        logger.info(f"Recording NEGATIVE feedback for conv_id: {conv_id}, msg_id: {msg_id}")
        record_feedback(conv_id, msg_id, "NEGATIVE", user_token=user_token)
        return "thumbs-up-button", "thumbs-down-button active"
    
    return no_update, no_update

# Callback 10: Toggle SQL query visibility
@app.callback(
    [Output({"type": "query-code", "index": MATCH}, "className"),
     Output({"type": "toggle-text", "index": MATCH}, "children")],
    [Input({"type": "toggle-query", "index": MATCH}, "n_clicks")],
    prevent_initial_call=True
)
def toggle_query_visibility(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return "query-code-container visible", "Hide code"
    return "query-code-container hidden", "Show code"

# Callback 11: Open welcome text modal
@app.callback(
    [Output("edit-welcome-modal", "is_open", allow_duplicate=True),
     Output("welcome-title-input", "value"),
     Output("welcome-description-input", "value"),
     Output("suggestion-1-input", "value"),
     Output("suggestion-2-input", "value"),
     Output("suggestion-3-input", "value"),
     Output("suggestion-4-input", "value")],
    [Input("edit-welcome-button", "n_clicks")],
    [State("welcome-title", "children"),
     State("welcome-description", "children"),
     State("suggestion-1-text", "children"),
     State("suggestion-2-text", "children"),
     State("suggestion-3-text", "children"),
     State("suggestion-4-text", "children")],
    prevent_initial_call=True
)
def open_modal(n_clicks, current_title, current_description, s1, s2, s3, s4):
    if not n_clicks:
        return [no_update] * 7
    return True, current_title, current_description, s1, s2, s3, s4

# Callback 12: Save/Close welcome text modal
@app.callback(
    [Output("welcome-title", "children", allow_duplicate=True),
     Output("welcome-description", "children", allow_duplicate=True),
     Output("suggestion-1-text", "children", allow_duplicate=True),
     Output("suggestion-2-text", "children", allow_duplicate=True),
     Output("suggestion-3-text", "children", allow_duplicate=True),
     Output("suggestion-4-text", "children", allow_duplicate=True),
     Output("edit-welcome-modal", "is_open", allow_duplicate=True)],
    [Input("save-welcome-text", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("welcome-title-input", "value"),
     State("welcome-description-input", "value"),
     State("suggestion-1-input", "value"), State("suggestion-2-input", "value"),
     State("suggestion-3-input", "value"), State("suggestion-4-input", "value"),
     State("welcome-title", "children"), State("welcome-description", "children"),
     State("suggestion-1-text", "children"), State("suggestion-2-text", "children"),
     State("suggestion-3-text", "children"), State("suggestion-4-text", "children")],
    prevent_initial_call=True
)
def handle_modal_actions(save_clicks, close_clicks,
                         new_title, new_description, s1, s2, s3, s4,
                         current_title, current_description,
                         current_s1, current_s2, current_s3, current_s4):
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * 7

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "close-modal":
        return [current_title, current_description, current_s1, current_s2, current_s3, current_s4, False]
    elif trigger_id == "save-welcome-text":
        title = new_title or DEFAULT_WELCOME_TITLE
        description = new_description or DEFAULT_WELCOME_DESCRIPTION
        suggestions = [
            s1 or DEFAULT_SUGGESTIONS[0], s2 or DEFAULT_SUGGESTIONS[1],
            s3 or DEFAULT_SUGGESTIONS[2], s4 or DEFAULT_SUGGESTIONS[3]
        ]
        return [title, description, *suggestions, False]
    return [no_update] * 7

# Callback 13: Generate insights
@app.callback(
    Output({"type": "insight-output", "index": MATCH}, "children"),
    Input({"type": "insight-button", "index": MATCH}, "n_clicks"),
    State({"type": "insight-button", "index": MATCH}, "id"),
    State("chat-history-store", "data"),
    State("session-store", "data"), # <-- ADDED
    prevent_initial_call=True
)
def generate_insights(n_clicks, btn_id, chat_history, session_data): # <-- ADDED session_data
    if not n_clicks:
        return None
    table_id = btn_id["index"]
    df = None
    
    # --- MODIFIED: Use session_data to find correct session ---
    current_session_index = session_data.get("current_session", 0)
    if current_session_index is None:
        current_session_index = 0
        
    if chat_history and current_session_index < len(chat_history):
        df_json = chat_history[current_session_index].get('dataframes', {}).get(table_id)
        if df_json:
            df = pd.read_json(df_json, orient='split')
    # --- END MODIFICATION ---
            
    if df is None:
        logger.error(f"Could not find DataFrame for id {table_id} in session {current_session_index}")
        return dcc.Markdown("Error: Could not retrieve data for insights.", className="insight-content")
    
    insights = call_llm_for_insights(df)
    return dcc.Markdown(insights, className="insight-content")

# --- NEW CALLBACK 14: Handle Dynamic Filters ---
@app.callback(
    Output({"type": "dynamic-graph", "index": MATCH}, "figure"),
    [Input({"type": "date-picker", "index": MATCH}, "start_date"),
     Input({"type": "date-picker", "index": MATCH}, "end_date"),
     Input({"type": "threshold-column", "index": MATCH}, "value"),
     Input({"type": "threshold-input", "index": MATCH}, "value")],
    [State({"type": "dynamic-graph", "index": MATCH}, "id"),
     State({"type": "query-store", "index": MATCH}, "data"),
     State("chat-history-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def handle_dynamic_filters(start_date, end_date, threshold_col, threshold_val,
                           graph_id, user_query, chat_history, session_data):
    
    ctx = callback_context
    if not ctx.triggered:
        logger.info("Filter callback triggered with no inputs.")
        return no_update

    df_id = graph_id["index"]
    
    # 1. Get the original DataFrame
    current_session_index = session_data.get("current_session", 0)
    if current_session_index is None: current_session_index = 0
    
    if not (chat_history and current_session_index < len(chat_history)):
        logger.error("Filter callback: Chat history not found.")
        return no_update
        
    df_json = chat_history[current_session_index].get('dataframes', {}).get(df_id)
    if not df_json:
        logger.error(f"Filter callback: DF_ID {df_id} not found in chat history.")
        return no_update
        
    df = pd.read_json(df_json, orient='split')
    
    # 2. Convert date columns (again, to be safe)
    date_col = None
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass
        if pd.api.types.is_datetime64_any_dtype(df[col]) and date_col is None:
            date_col = col
            
    filtered_df = df.copy()

    # 3. Apply filters
    try:
        # Date filter
        if start_date and end_date and date_col:
            filtered_df = filtered_df[
                (filtered_df[date_col] >= pd.to_datetime(start_date)) &
                (filtered_df[date_col] <= pd.to_datetime(end_date))
            ]

        # Threshold filter
        if threshold_col and (threshold_val is not None):
            filtered_df = filtered_df[
                filtered_df[threshold_col] >= float(threshold_val)
            ]
    except Exception as e:
        logger.error(f"Error applying filters: {e}")
        # Continue with partially filtered or unfiltered df
        pass

    if filtered_df.empty:
        return {"layout": {"title": "No data matches your filters."}}

    # 4. Regenerate visual spec with the filtered data
    visual_spec_raw = get_visual_spec(filtered_df, user_query)
    visual_spec_clean = visual_spec_raw.strip().replace("json","").replace("```","").replace("'",'"').replace("\\n","").replace("\r","")
    
    try:
        visual_spec = json.loads(visual_spec_clean)
        return visual_spec
    except Exception as e:
        logger.error(f"Failed to parse visual spec in filter callback: {e}")
        return {"layout": {"title": f"Error updating visual: {e}"}}

# --- NEW CALLBACK 15: Handle Chart Selection ---
@app.callback(
    Output({"type": "selection-info", "index": MATCH}, "children"),
    Input({"type": "dynamic-graph", "index": MATCH}, "selectedData"),
    [State({"type": "dynamic-graph", "index": MATCH}, "id"),
     State("chat-history-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def handle_chart_selection(selectedData, graph_id, chat_history, session_data):
    if not selectedData or not selectedData.get('points'):
        return html.P("Use the Box Select or Lasso Select tool on the chart to see details here.", style={"fontStyle": "italic", "color": "#888"})

    df_id = graph_id["index"]
    
    # 1. Get the original DataFrame
    current_session_index = session_data.get("current_session", 0)
    if current_session_index is None: current_session_index = 0
    
    if not (chat_history and current_session_index < len(chat_history)):
        return no_update
        
    df_json = chat_history[current_session_index].get('dataframes', {}).get(df_id)
    if not df_json:
        return no_update
        
    df = pd.read_json(df_json, orient='split')

    # 2. Get selected data
    try:
        selected_indices = [p['pointIndex'] for p in selectedData['points']]
        selected_df = df.iloc[selected_indices]
    except (KeyError, IndexError):
        return html.P("Could not read selection data.", style={"color": "red"})

    if selected_df.empty:
        return html.P("No data selected.")

    # 3. Create summary
    num_points = len(selected_df)
    summary_children = [html.H5(f"Selection Summary ({num_points} points):")]
    
    numeric_cols = selected_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        col_sum = selected_df[col].sum()
        col_mean = selected_df[col].mean()
        summary_children.append(
            html.Div(f"Total {col}: {col_sum:,.2f} (Avg: {col_mean:,.2f})", style={"fontSize": "14px"})
        )
        
    return html.Div(summary_children, style={"padding": "10px"})

# --- NEW CALLBACK 16: Set User Initial on Load ---
@app.callback(
    Output("user-avatar", "children"),
    Input("initial-load-trigger", "data"),
    prevent_initial_call=False # <-- This makes it run on load
)
def set_user_initial(load_trigger):
    try:
        _, initial = get_user_info_from_header()
        return initial
    except Exception as e:
        logger.warning(f"Error setting user initial: {e}")
        return "Y"
# --- END NEW CALLBACKS ---


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
