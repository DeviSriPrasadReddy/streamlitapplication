import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context, no_update, clientside_callback, dash_table
import dash_bootstrap_components as dbc
import json
import logging
import dash.flask
from genieroom import genie_query, record_feedback
import pandas as pd
import os
from dotenv import load_dotenv
import sqlparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# --- NEW IMPORTS ---
import plotly.express as px
import plotly.graph_objects as go
# --- END NEW IMPORTS ---

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_user_token_from_header():
    """Reads the user token from the request headers."""
    try:
        header_name = 'X-Databricks-User-Token' # Change if needed
        token = dash.flask.request.headers.get(header_name)
        if token:
            return token
    except Exception:
        pass
    logger.warning("Could not find user token in request headers. Will use fallback.")
    return None

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    # Suppress callback exceptions for the new dynamic components
    suppress_callback_exceptions=True 
)
server = app.server

# ... (Default welcome text/suggestions are unchanged) ...
DEFAULT_WELCOME_TITLE = "Supply Chain Optimization"
DEFAULT_WELCOME_DESCRIPTION = "Analyze your Supply Chain Performance leveraging AI/BI Dashboard. Deep dive into your data and metrics."
DEFAULT_SUGGESTIONS = [
    "What tables are there and how are they connected? Give me a short summary.",
    "Which distribution center has the highest chance of being a bottleneck?",
    "Explain the dataset",
    "What was the demand for our products by week in 2024?"
]


# Define the layout
app.layout = html.Div([
    # ... (Top navigation bar is unchanged) ...
    html.Div([
        html.Div([
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
            html.Div("Y", className="user-avatar"),
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
    
    # ... (Main content area is unchanged) ...
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([html.Div([
                    html.Div(className="genie-logo")
                ], className="genie-logo-container")],
                className="genie-logo-container-header"),
                
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
                    
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Customize Welcome Message")),
                        dbc.ModalBody([
                            html.Div([
                                html.Label("Welcome Title", className="modal-label"),
                                dbc.Input(id="welcome-title-input", type="text", placeholder="Enter a title...", className="modal-input"),
                                html.Small("This title appears at the top of your welcome screen", className="text-muted d-block mt-1")
                            ], className="modal-input-group"),
                            html.Div([
                                html.Label("Welcome Description", className="modal-label"),
                                dbc.Textarea(id="welcome-description-input", placeholder="Enter a description...", className="modal-input", style={"height": "80px"}),
                                html.Small("This description appears below the title", className="text-muted d-block mt-1")
                            ], className="modal-input-group"),
                            html.Div([
                                html.Label("Suggestion Questions", className="modal-label"),
                                html.Small("Customize the four suggestion questions", className="text-muted d-block mb-3"),
                                dbc.Input(id="suggestion-1-input", type="text", placeholder="First suggestion", className="modal-input mb-2"),
                                dbc.Input(id="suggestion-2-input", type="text", placeholder="Second suggestion", className="modal-input mb-2"),
                                dbc.Input(id="suggestion-3-input", type="text", placeholder="Third suggestion", className="modal-input mb-2"),
                                dbc.Input(id="suggestion-4-input", type="text", placeholder="Fourth suggestion", className="modal-input")
                            ], className="modal-input-group")
                        ]),
                        dbc.ModalFooter([
                            dbc.Button("Cancel", id="close-modal", className="modal-button", color="light"),
                            dbc.Button("Save Changes", id="save-welcome-text", className="modal-button-primary", color="primary")
                        ])
                    ], id="edit-welcome-modal", is_open=False, size="lg", backdrop="static"),
                    
                    html.Div([
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[0], className="suggestion-text", id="suggestion-1-text")
                        ], id="suggestion-1", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[1], className="suggestion-text", id="suggestion-2-text")
                        ], id="suggestion-2", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[2], className="suggestion-text", id="suggestion-3-text")
                        ], id="suggestion-3", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div(DEFAULT_SUGGESTIONS[3], className="suggestion-text", id="suggestion-4-text")
                        ], id="suggestion-4", className="suggestion-button")
                    ], className="suggestion-buttons")
                ], id="welcome-container", className="welcome-container visible"),
                
                html.Div([], id="chat-messages", className="chat-messages"),
            ], id="chat-content", className="chat-content"),
            
            html.Div([
                html.Div([
                    dcc.Input(id="chat-input-fixed", placeholder="Ask your question...", className="chat-input", type="text", disabled=False),
                    html.Div([
                        html.Button(id="send-button-fixed", className="input-button send-button", disabled=False)
                    ], className="input-buttons-right"),
                    html.Div("You can only submit one query at a time", id="query-tooltip", className="query-tooltip hidden")
                ], id="fixed-input-container", className="fixed-input-container"),
                html.Div("Always review the accuracy of responses.", className="disclaimer-fixed")
            ], id="fixed-input-wrapper", className="fixed-input-wrapper"),
        ], id="chat-container", className="chat-container"),
    ], id="main-content", className="main-content"),
    
    html.Div(id='dummy-output'),
    dcc.Store(id="chat-trigger", data={"trigger": False, "message": "", "conversation_id": None}),
    dcc.Store(id="chat-history-store", data=[], storage_type='session'),
    dcc.Store(id="query-running-store", data=False),
    dcc.Store(id="session-store", data={"current_session": None, "conversation_id": None}, storage_type='session')
])

def format_sql_query(sql_query):
    return sqlparse.format(
        sql_query, keyword_case='upper', reindent=True, indent_width=2
    )

# --- MODIFICATION: Function now accepts user_token ---
def call_llm_for_insights(df, prompt=None, user_token: str = None):
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
        # --- MODIFICATION: Initialize client based on token ---
        if user_token:
            client = WorkspaceClient(host=os.getenv("DATABRICKS_HOST"), token=user_token)
            logger.info("call_llm_for_insights: Using user token.")
        else:
            client = WorkspaceClient() # Fallback to SP
            logger.warning("call_llm_for_insights: No user token, falling back to Service Principal.")
        # --- END MODIFICATION ---
        
        # --- !!! CHECK YOUR PAYLOAD ---
        response = client.serving_endpoints.query(
            os.getenv("SERVING_ENDPOINT_NAME"),
            messages=[ChatMessage(content=full_prompt, role=ChatMessageRole.USER)],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return f"Error generating insights: {str(e)}"

# --- MODIFICATION: Function now accepts user_token ---
def get_followup_questions(user_query: str, bot_response: str, user_token: str = None) -> dict:
    SUGGESTION_ENDPOINT_NAME = os.environ.get("SUGGESTION_ENDPOINT_NAME")
    if not SUGGESTION_ENDPOINT_NAME:
        logger.warning("SUGGESTION_ENDPOINT_NAME not set. Skipping follow-up questions.")
        return {}
    prompt = f"""
    Given a user's question and a chatbot's answer, generate one "better" version of the user's question and two relevant follow-up questions.
    Return ONLY a single valid JSON object with the keys "better_prompt", "followup1", and "followup2".
    User Question: "{user_query}"
    Chatbot Answer: "{bot_response[:1000]}"
    JSON:
    """
    try:
        # --- MODIFICATION: Initialize client based on token ---
        if user_token:
            client = WorkspaceClient(host=os.getenv("DATABRICKS_HOST"), token=user_token)
            logger.info("get_followup_questions: Using user token.")
        else:
            client = WorkspaceClient() # Fallback to SP
            logger.warning("get_followup_questions: No user token, falling back to Service Principal.")
        # --- END MODIFICATION ---
        
        # --- !!! CHECK YOUR PAYLOAD ---
        payload = {
            "prompt": prompt, "max_tokens": 200, "temperature": 0.5
        }
        response = client.serving_endpoints.query(
            SUGGESTION_ENDPOINT_NAME,
            request=payload
        )
        if "predictions" in response and response.predictions:
            json_str = response.predictions[0].strip().replace("```json", "").replace("```", "")
            data = json.loads(json_str)
            return data
        else:
            logger.warning(f"Unexpected response from suggestion endpoint: {response}")
            return {}
    except Exception as e:
        logger.error(f"Error getting follow-up questions: {e}")
        return {}


def create_smart_chart(df: pd.DataFrame, chart_id: str) -> html.Div:
    # ... (This function is unchanged, it runs locally) ...
    try:
        plot_df = df.copy()
        numerics = plot_df.select_dtypes(include=['number']).columns.tolist()
        objects = plot_df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = []
        for col in objects:
            try:
                converted_col = pd.to_datetime(plot_df[col], errors='coerce')
                if converted_col.notna().sum() / len(plot_df) > 0.5:
                    plot_df[col] = converted_col
                    date_cols.append(col)
            except Exception:
                pass
        objects = [col for col in objects if col not in date_cols]
        fig = None
        
        if date_cols and numerics:
            date_col, num_col = date_cols[0], numerics[0]
            plot_df = plot_df.sort_values(by=date_col)
            fig = px.line(plot_df, x=date_col, y=num_col, title=f"{num_col} over {date_col}", markers=True)
            logger.info(f"Smart Chart: Detected Time Series, creating Line Chart ({num_col} vs {date_col})")
        elif objects and numerics:
            cat_col, num_col = objects[0], numerics[0]
            unique_count = plot_df[cat_col].nunique()
            if unique_count <= 8:
                fig = px.pie(plot_df, names=cat_col, values=num_col, title=f"Distribution of {num_col} by {cat_col}")
                logger.info(f"Smart Chart: Detected few categories, creating Pie Chart ({num_col} vs {cat_col})")
            elif unique_count <= 50:
                plot_df = plot_df.sort_values(by=num_col, ascending=False)
                fig = px.bar(plot_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                logger.info(f"Smart Chart: Detected categories, creating Bar Chart ({num_col} vs {cat_col})")
        elif len(numerics) >= 2:
            num_col_1, num_col_2 = numerics[0], numerics[1]
            fig = px.scatter(df, x=num_col_1, y=num_col_2, title=f"{num_col_2} vs. {num_col_1}",
                             trendline="ols", trendline_color_override="red")
            logger.info(f"Smart Chart: Detected 2+ numerics, creating Scatter Plot ({num_col_2} vs {num_col_1})")
        
        if fig:
            fig.update_layout(
                template="plotly_white", title_x=0.5,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return dcc.Graph(id=chart_id, figure=fig)
        
        logger.info("Smart Chart: No suitable chart type found for the data.")
        return None
    except Exception as e:
        logger.error(f"Error creating smart chart: {e}")
        return None


# Callback 1: Handle inputs and show thinking indicator
@app.callback(
    # ... (Outputs are unchanged) ...
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-input-fixed", "value", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    # ... (Inputs and States are unchanged) ...
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
     State({"type": "followup-suggestion", "index": ALL}, "id")
    ],
    prevent_initial_call=True
)
def handle_all_inputs(s1_clicks, s2_clicks, s3_clicks, s4_clicks, send_clicks, submit_clicks,
                      followup_clicks,
                      s1_text, s2_text, s3_text, s4_text, input_value, current_messages,
                      welcome_class, current_chat_list, chat_history, session_data,
                      followup_ids):
    # ... (This function is unchanged) ...
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
            trigger_id_dict = json.loads(trigger_id_str)
            user_input = trigger_id_dict.get("text")
        except: pass
    elif trigger_id_str in suggestion_map:
        user_input = suggestion_map[trigger_id_str]
    else:
        user_input = input_value
    if not user_input:
        return [no_update] * 8
    user_message = html.Div([
        html.Div([html.Div("Y", className="user-avatar"), html.Span("You", className="model-name")], className="user-info"),
        html.Div(user_input, className="message-text")
    ], className="user-message message")
    updated_messages = (current_messages or []) + [user_message]
    thinking_indicator = html.Div([
        html.Div([html.Span(className="spinner"), html.Span("Thinking...")], className="thinking-indicator")
    ], className="bot-message message")
    updated_messages.append(thinking_indicator)
    current_session_index = session_data.get("current_session")
    current_conv_id = session_data.get("conversation_id")
    if current_session_index is None:
        current_session_index = 0
    chat_history = chat_history or []
    if current_session_index < len(chat_history):
        chat_history[current_session_index]["messages"] = updated_messages
        chat_history[current_session_index]["queries"].append(user_input)
    else:
        chat_history.insert(0, {
            "session_id": current_session_index,
            "backend_conversation_id": current_conv_id,
            "queries": [user_input],
            "messages": updated_messages
        })
    updated_chat_list = []
    for i, session in enumerate(chat_history):
        first_query = session["queries"][0]
        is_active = (i == current_session_index) 
        updated_chat_list.append(
            html.Div(
                first_query,
                className=f"chat-item{' active' if is_active else ''}",
                id={"type": "chat-item", "index": i}
            )
        )
    trigger_data = {
        "trigger": True, 
        "message": user_input,
        "conversation_id": current_conv_id
    }
    updated_session_data = {
        "current_session": current_session_index, 
        "conversation_id": current_conv_id
    }
    return (updated_messages, "", "welcome-container hidden",
            trigger_data, True,
            updated_chat_list, chat_history, updated_session_data)


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
    
    user_token = get_user_token_from_header()
    
    try:
        conv_id, msg_id, response, query_text = genie_query(
            user_input, conversation_id, user_token=user_token
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
        response_text_for_context = ""
        chart_section = None
        
        if isinstance(response, str):
            content = dcc.Markdown(response, className="message-text")
            response_text_for_context = response
        else:
            df = pd.DataFrame(response)
            df_id = f"table-{len(chat_history)}-{len(current_messages)}"
            chart_index = f"chart-{len(chat_history)}-{len(current_messages)}"
            
            chart_component = create_smart_chart(df, chart_id={"type": "chart-graph", "index": chart_index})
            
            if chart_component:
                # --- MODIFICATION: Make chart visible by default ---
                chart_section = html.Div([
                    html.Button([
                        html.Span("Hide Chart", id={"type": "toggle-chart-text", "index": chart_index})
                    ], id={"type": "toggle-chart", "index": chart_index}, className="toggle-query-button", n_clicks=0),
                    html.Div(
                        chart_component,
                        id={"type": "chart-container", "index": chart_index},
                        className="query-code-container visible" # <-- CHANGED from 'hidden'
                    )
                ], className="query-section")
                # --- END MODIFICATION ---

            if chat_history and len(chat_history) > 0:
                chat_history[0].setdefault('dataframes', {})[df_id] = df.to_json(orient='split')
            else:
                chat_history = [{"dataframes": {df_id: df.to_json(orient='split')}}]
            
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
            
            query_section = None
            if query_text is not None:
                formatted_sql = format_sql_query(query_text)
                query_index = f"sql-{len(chat_history)}-{len(current_messages)}"
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
            
            content = html.Div([
                chart_section if chart_section else None,
                html.Div([data_table], style={'marginBottom': '20px', 'paddingRight': '5px'}),
                query_section if query_section else None,
                insight_button, insight_output,
            ])
            response_text_for_context = f"A table was returned for the query: {query_text or '...'} "
        
        pill_style = {"backgroundColor": "#f0f0f0", "border": "1px solid #ddd", "borderRadius": "16px", "padding": "6px 12px", "margin": "4px", "fontSize": "13px", "cursor": "pointer", "display": "block", "textAlign": "left", "width": "fit-content", "maxWidth": "100%", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap", "lineHeight": "1.4"}
        prefix_style = {"fontWeight": "600", "marginRight": "5px"}
        container_style = {"paddingTop": "10px", "marginTop": "10px", "borderTop": "1px solid #eee"}
        suggestion_div = html.Div(style=container_style)
        try:
            # --- MODIFICATION: Pass user_token ---
            suggestions = get_followup_questions(user_input, response_text_for_context, user_token=user_token)
            # --- END MODIFICATION ---
            
            suggestion_elements = []
            if suggestions.get("better_prompt"):
                suggestion_elements.append(
                    html.Button([
                        html.Span("💡 Better way to ask: ", style=prefix_style),
                        html.Span(suggestions["better_prompt"])
                    ], id={"type": "followup-suggestion", "index": 0, "text": suggestions["better_prompt"]}, style=pill_style)
                )
            if suggestions.get("followup1"):
                suggestion_elements.append(
                    html.Button([
                        html.Span("Relevant question: ", style=prefix_style),
                        html.Span(suggestions["followup1"])
                    ], id={"type": "followup-suggestion", "index": 1, "text": suggestions["followup1"]}, style=pill_style)
                )
            if suggestions.get("followup2"):
                suggestion_elements.append(
                    html.Button([
                        html.Span("Relevant question: ", style=prefix_style),
                        html.Span(suggestions["followup2"])
                    ], id={"type": "followup-suggestion", "index": 2, "text": suggestions["followup2"]}, style=pill_style)
                )
            if suggestion_elements:
                suggestion_div = html.Div(suggestion_elements, style=container_style)
            else:
                suggestion_div = html.Div()
        except Exception as e:
            logger.error(f"Failed to generate follow-up suggestions: {e}")
            suggestion_div = html.Div()

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
                suggestion_div
            ], className="message-content")
        ], className="bot-message message")
        
        updated_messages = (current_messages or [])[:-1] + [bot_response]
        
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = updated_messages
            chat_history[0]["backend_conversation_id"] = conv_id

        return updated_messages, chat_history, {"trigger": False, "message": ""}, False, new_session_data
        
    except Exception as e:
        logger.error(f"Error in get_model_response: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again later."
        error_response = html.Div([
            html.Div([html.Div(className="model-avatar"), html.Span("Genie", className="model-name")], className="model-info"),
            html.Div([html.Div(error_msg, className="message-text")], className="message-content")
        ], className="bot-message message")
        updated_messages = (current_messages or [])[:-1] + [error_response]
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = updated_messages
        return updated_messages, chat_history, {"trigger": False, "message": ""}, False, no_update

# ... (Callbacks 3-8 are unchanged) ...
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
        for i, item in enumerate(chat_list):
            updated_chat_list.append(
                html.Div(item["props"]["children"], className="chat-item", id={"type": "chat-item", "index": i})
            )
    else:
        updated_chat_list = no_update
    return ("welcome-container visible", [], {"trigger": False, "message": ""}, 
            False, new_session_data, updated_chat_list)

# Callback 7: Hide welcome on chat
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
    
    user_token = get_user_token_from_header()
    
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
    [State({"type": "query-code", "index": MATCH}, "className")],
    prevent_initial_call=True
)
def toggle_query_visibility(n_clicks, current_class):
    if n_clicks and "hidden" in current_class:
        return "query-code-container visible", "Hide code"
    return "query-code-container hidden", "Show code"


# --- MODIFIED CALLBACK: Toggle Chart Visibility ---
@app.callback(
    [Output({"type": "chart-container", "index": MATCH}, "className"),
     Output({"type": "toggle-chart-text", "index": MATCH}, "children")],
    [Input({"type": "toggle-chart", "index": MATCH}, "n_clicks")],
    [State({"type": "chart-container", "index": MATCH}, "className")],
    prevent_initial_call=True
)
def toggle_chart_visibility(n_clicks, current_class):
    # This logic is now correct for the new default "visible" state
    if n_clicks and "visible" in current_class:
        # It was visible, so n_clicks=1, 3, 5... will hide it
        return "query-code-container hidden", "Show Chart"
    else:
        # It was hidden (or n_clicks=0), so n_clicks=2, 4, 6... will show it
        return "query-code-container visible", "Hide Chart"
# --- END MODIFIED CALLBACK ---


# ... (Callbacks for welcome text modal are unchanged) ...
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
    State("session-store", "data"),
    prevent_initial_call=True
)
def generate_insights(n_clicks, btn_id, chat_history, session_data):
    if not n_clicks:
        return None
    
    table_id = btn_id["index"]
    df = None
    
    current_session_index = session_data.get("current_session", 0)
    if not chat_history or current_session_index >= len(chat_history):
        logger.error(f"Could not find session {current_session_index} in chat history.")
        return dcc.Markdown("Error: Could not retrieve data for insights.", className="insight-content")

    df_json = chat_history[current_session_index].get('dataframes', {}).get(table_id)
    if df_json:
        df = pd.read_json(df_json, orient='split')
            
    if df is None:
        logger.error(f"Could not find DataFrame for id {table_id} in session {current_session_index}")
        return dcc.Markdown("Error: Could not retrieve data for insights.", className="insight-content")
    
    # --- MODIFICATION: Pass user_token ---
    user_token = get_user_token_from_header()
    insights = call_llm_for_insights(df, user_token=user_token)
    # --- END MODIFICATION ---
    
    return dcc.Markdown(insights, className="insight-content")


if __name__ == "__main__":
    # Note: app.run_server is for development.
    # For production, use Gunicorn: `gunicorn app:server`
    app.run_server(debug=True, host='0.0.0.0', port=8050)

