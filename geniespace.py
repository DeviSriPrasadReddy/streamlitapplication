import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import backoff
import uuid
# Import your Service Principal token minter
from token_minter import TokenMinter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load environment variables
SPACE_ID = os.environ.get("SPACE_ID")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")

# --- BACKUP TOKEN GENERATOR ---
# Initialize the Service Principal token minter as a fallback
sp_token_minter = TokenMinter(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    host=DATABRICKS_HOST
)


class GenieClient:
    """
    GenieClient now accepts an optional 'user_token'.
    If 'user_token' is provided, it's used for all API calls.
    If it's None, it falls back to the Service Principal token.
    """
    def __init__(self, host: str, space_id: str, user_token: Optional[str] = None):
        self.host = host
        self.space_id = space_id
        self.user_token = user_token  # Store the user's token
        self.base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
        self.update_headers() # Call update_headers *after* user_token is set
    
    def update_headers(self) -> None:
        """Update headers with user token if available, otherwise fallback to SP token"""
        
        token = ""
        if self.user_token:
            # --- PRIMARY ---
            # Use the provided user token
            token = self.user_token
            logger.info("Using user-level token for API call.")
        else:
            # --- FALLBACK ---
            # Use the service principal token
            logger.warning("No user token provided, falling back to Service Principal token.")
            token = sp_token_minter.get_token()

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2,
        on_backoff=lambda details: logger.warning(f"API retry {details['tries']} in {details['wait']:.1f}s")
    )
    def start_conversation(self, question: str) -> Dict[str, Any]:
        """Start a new conversation with the given question"""
        self.update_headers()  # Refresh token before API call (will use user or SP token)
        url = f"{self.base_url}/start-conversation"
        payload = {"content": question}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2,
        on_backoff=lambda details: logger.warning(f"API retry {details['tries']} in {details['wait']:.1f}s")
    )
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Send a follow-up message to an existing conversation"""
        self.update_headers()
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": message}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2,
        on_backoff=lambda details: logger.warning(f"API retry {details['tries']} in {details['wait']:.1f}s")
    )
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        """Get the details of a specific message"""
        self.update_headers()
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2,
        on_backoff=lambda details: logger.warning(f"API retry {details['tries']} in {details['wait']:.1f}s")
    )
    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Get the query result using the attachment_id endpoint"""
        self.update_headers()
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        data_array = result.get('statement_response', {}).get('result', {}).get('data_array', [])
        return {
            'data_array': data_array,
            'schema': result.get('statement_response', {}).get('manifest', {}).get('schema', {})
        }

    # (execute_query method removed for brevity, assuming it's similar to others)

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=3,
        on_backoff=lambda details: logger.warning(f"Feedback API retry {details['tries']} in {details['wait']:.1f}s")
    )
    def send_feedback(self, conversation_id: str, message_id: str, feedback_type: str) -> Dict[str, Any]:
        """Send feedback for a specific message to the backend."""
        self.update_headers()
        
        # --- !!!  ACTION REQUIRED  !!! ---
        # Update this URL to your actual feedback endpoint
        feedback_url = f"{self.base_url}/feedback" 
        
        # --- !!!  ACTION REQUIRED  !!! ---
        # Update this payload to match your API's expected format
        payload = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "feedback_type": feedback_type  # e.g., "POSITIVE" or "NEGATIVE"
        }
        
        logger.info(f"Sending feedback to {feedback_url}: {payload}")
        response = requests.post(feedback_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def wait_for_message_completion(self, conversation_id: str, message_id: str, timeout: int = 300, poll_interval: int = 2) -> Dict[str, Any]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            message = self.get_message(conversation_id, message_id)
            status = message.get("status")
            if status in ["COMPLETED", "ERROR", "FAILED"]:
                return message
            time.sleep(poll_interval)
        raise TimeoutError(f"Message processing timed out after {timeout} seconds")

# --- HELPER FUNCTIONS (Modified to pass user_token) ---

def start_new_conversation(question: str, user_token: Optional[str] = None) -> Tuple[str, str, Union[str, pd.DataFrame], Optional[str]]:
    """Starts a new conversation, passing the user_token to the client."""
    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID,
        user_token=user_token  # Pass the token
    )
    try:
        response = client.start_conversation(question)
        conversation_id = response.get("conversation_id")
        message_id = response.get("message_id")
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        result, query_text = process_genie_response(client, conversation_id, message_id, complete_message)
        return conversation_id, message_id, result, query_text
    except Exception as e:
        logger.error(f"Error starting new conversation: {str(e)}")
        return None, None, f"Sorry, an error occurred: {str(e)}. Please try again.", None

def continue_conversation(conversation_id: str, question: str, user_token: Optional[str] = None) -> Tuple[str, Union[str, pd.DataFrame], Optional[str]]:
    """Continues a conversation, passing the user_token to the client."""
    logger.info(f"Continuing conversation {conversation_id}...")
    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID,
        user_token=user_token # Pass the token
    )
    try:
        response = client.send_message(conversation_id, question)
        message_id = response.get("message_id")
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        result, query_text = process_genie_response(client, conversation_id, message_id, complete_message)
        return message_id, result, query_text
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            return None, "Sorry, the system is currently experiencing high demand. Please try again in a few moments.", None
        elif "Conversation not found" in str(e):
            logger.warning(f"Conversation {conversation_id} not found. A new session will be required.")
            return None, "Sorry, the previous conversation has expired. Please try your query again to start a new conversation.", None
        else:
            logger.error(f"Error continuing conversation: {str(e)}")
            return None, f"Sorry, an error occurred: {str(e)}", None

def process_genie_response(client: GenieClient, conversation_id: str, message_id: str, complete_message: Dict[str, Any]) -> Tuple[Union[str, pd.DataFrame], Optional[str]]:
    """Processes the completed message (no changes needed here)."""
    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")
        if "text" in attachment and "content" in attachment["text"]:
            return attachment["text"]["content"], None
        elif "query" in attachment:
            query_text = attachment.get("query", {}).get("query", "")
            query_result = client.get_query_result(conversation_id, message_id, attachment_id)
            data_array = query_result.get('data_array', [])
            schema = query_result.get('schema', {})
            columns = [col.get('name') for col in schema.get('columns', [])]
            if data_array:
                if not columns and data_array:
                    columns = [f"column_{i}" for i in range(len(data_array[0]))]
                df = pd.DataFrame(data_array, columns=columns)
                return df, query_text
    
    if 'content' in complete_message:
        return complete_message.get('content', ''), None
    return "No response available", None

# --- MAIN ENTRYPOINT (Modified to pass user_token) ---

def genie_query(question: str, conversation_id: Optional[str] = None, user_token: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Union[str, pd.DataFrame], Optional[str]]:
    """
    Main entry point for querying Genie.
    Passes the user_token to the conversation handlers.
    """
    try:
        if conversation_id is None:
            logger.info("Starting new conversation...")
            conv_id, msg_id, result, query_text = start_new_conversation(question, user_token=user_token)
            if conv_id is None:
                 return None, None, result, None
            return conv_id, msg_id, result, query_text
        else:
            logger.info(f"Continuing conversation {conversation_id}...")
            msg_id, result, query_text = continue_conversation(conversation_id, question, user_token=user_token)
            if msg_id is None and "expired" in str(result):
                return None, None, result, None
            return conversation_id, msg_id, result, query_text
    except Exception as e:
        logger.error(f"Error in genie_query: {str(e)}. Please try again.")
        return conversation_id, None, f"Sorry, an error occurred: {str(e)}. Please try again.", None

# --- FEEDBACK ENTRYPOINT (Modified to pass user_token) ---

def record_feedback(conversation_id: str, message_id: str, feedback_type: str, user_token: Optional[str] = None) -> Tuple[bool, str]:
    """High-level helper to record feedback, passing the user_token."""
    if not all([conversation_id, message_id, feedback_type]):
        logger.warning("Feedback attempted with missing IDs.")
        return False, "Missing conversation_id or message_id"
        
    logger.info(f"Recording feedback for message {message_id} in conversation {conversation_id}: {feedback_type}")
    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID,
        user_token=user_token # Pass the token
    )
    try:
        response = client.send_feedback(conversation_id, message_id, feedback_type.upper())
        logger.info(f"Feedback API response: {response}")
        return True, "Feedback recorded successfully"
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        return False, f"Error recording feedback: {str(e)}"
