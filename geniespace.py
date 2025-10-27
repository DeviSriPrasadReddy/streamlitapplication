import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import backoff
import uuid
# Assuming token_minter.py exists in the same directory
from token_minter import TokenMinter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load environment variables
SPACE_ID = os.environ.get("SPACE_ID")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")

token_minter = TokenMinter(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    host=DATABRICKS_HOST
)


class GenieClient:
    def __init__(self, host: str, space_id: str):
        self.host = host
        self.space_id = space_id
        self.update_headers()
        
        self.base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
    
    def update_headers(self) -> None:
        """Update headers with fresh token from token_minter"""
        self.headers = {
            "Authorization": f"Bearer {token_minter.get_token()}",
            "Content-Type": "application/json"
        }
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def start_conversation(self, question: str) -> Dict[str, Any]:
        """Start a new conversation with the given question"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/start-conversation"
        payload = {"content": question}
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Send a follow-up message to an existing conversation"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": message}
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        """Get the details of a specific message"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Get the query result using the attachment_id endpoint"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        # Extract data_array from the correct nested location
        data_array = []
        if 'statement_response' in result:
            if 'result' in result['statement_response']:
                data_array = result['statement_response']['result'].get('data_array', [])
        
        return {
                    'data_array': data_array,
                    'schema': result.get('statement_response', {}).get('manifest', {}).get('schema', {})
                }

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def execute_query(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Execute a query using the attachment_id endpoint"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/execute-query"
        
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    # --- NEW FUNCTION FOR FEEDBACK (Request 2) ---
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3, # Fail faster on feedback
        on_backoff=lambda details: logger.warning(
            f"Feedback API request failed. Retrying in {details['wait']:.2f} seconds."
        )
    )
    def send_feedback(self, conversation_id: str, message_id: str, feedback_type: str) -> Dict[str, Any]:
        """Send feedback for a specific message to the backend."""
        self.update_headers()
        
        # !!! ATTENTION: ASSUMPTION !!!
        # Please update this URL to your actual feedback endpoint
        # Example: "https://<host>/api/2.0/genie/spaces/<space_id>/feedback"
        feedback_url = f"{self.base_url}/feedback" 
        
        # !!! ATTENTION: ASSUMPTION !!!
        # Please update this payload to match your API's expected format
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
        """
        Wait for a message to reach a terminal state (COMPLETED, ERROR, etc.).
        """
        
        start_time = time.time()
        attempt = 1
        
        while time.time() - start_time < timeout:
            
            message = self.get_message(conversation_id, message_id)
            status = message.get("status")
            
            if status in ["COMPLETED", "ERROR", "FAILED"]:
                return message
                
            time.sleep(poll_interval)
            attempt += 1
            
        raise TimeoutError(f"Message processing timed out after {timeout} seconds")

# --- MODIFIED FUNCTION (Request 1) ---
def start_new_conversation(question: str) -> Tuple[str, str, Union[str, pd.DataFrame], Optional[str]]:
    """
    Start a new conversation with Genie.
    
    Returns:
        Tuple containing:
        - conversation_id: The new conversation ID
        - message_id: The ID of the bot's response message
        - response: Either text or DataFrame response
        - query_text: SQL query text if applicable, otherwise None
    """
    
    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID
    )
    
    try:
        # Start a new conversation
        response = client.start_conversation(question)
        conversation_id = response.get("conversation_id")
        message_id = response.get("message_id") # This is the ID for the bot's message
        
        # Wait for the message to complete
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        
        # Process the response
        result, query_text = process_genie_response(client, conversation_id, message_id, complete_message)
        
        # Return all IDs
        return conversation_id, message_id, result, query_text
        
    except Exception as e:
        logger.error(f"Error starting new conversation: {str(e)}")
        return None, None, f"Sorry, an error occurred: {str(e)}. Please try again.", None

# --- MODIFIED FUNCTION (Request 1) ---
def continue_conversation(conversation_id: str, question: str) -> Tuple[str, Union[str, pd.DataFrame], Optional[str]]:
    """
    Send a follow-up message in an existing conversation.
    
    Returns:
        Tuple containing:
        - message_id: The ID of the bot's response message
        - response: Either text or DataFrame response
        - query_text: SQL query text if applicable, otherwise None
    """
    logger.info(f"Continuing conversation {conversation_id} with question: {question[:30]}...")
    
    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID
    )
    
    try:
        # Send follow-up message in existing conversation
        response = client.send_message(conversation_id, question)
        message_id = response.get("message_id") # This is the ID for the bot's message
        
        # Wait for the message to complete
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        
        # Process the response
        result, query_text = process_genie_response(client, conversation_id, message_id, complete_message)
        
        # Return new message ID and results
        return message_id, result, query_text
        
    except Exception as e:
        # Handle specific errors
        if "429" in str(e) or "Too Many Requests" in str(e):
            return None, "Sorry, the system is currently experiencing high demand. Please try again in a few moments.", None
        elif "Conversation not found" in str(e):
            # This is a critical error for session management
            logger.warning(f"Conversation {conversation_id} not found. A new session will be required.")
            return None, "Sorry, the previous conversation has expired. Please try your query again to start a new conversation.", None
        else:
            logger.error(f"Error continuing conversation: {str(e)}")
            return None, f"Sorry, an error occurred: {str(e)}", None

def process_genie_response(client, conversation_id, message_id, complete_message) -> Tuple[Union[str, pd.DataFrame], Optional[str]]:
    """
    Process the response from Genie
    """
    # Check attachments first
    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")
        
        # If there's text content in the attachment, return it
        if "text" in attachment and "content" in attachment["text"]:
            return attachment["text"]["content"], None
        
        # If there's a query, get the result
        elif "query" in attachment:
            query_text = attachment.get("query", {}).get("query", "")
            query_result = client.get_query_result(conversation_id, message_id, attachment_id)
            
            data_array = query_result.get('data_array', [])
            schema = query_result.get('schema', {})
            columns = [col.get('name') for col in schema.get('columns', [])]
            
            # If we have data, return as DataFrame
            if data_array:
                # If no columns from schema, create generic ones
                if not columns and data_array and len(data_array) > 0:
                    columns = [f"column_{i}" for i in range(len(data_array[0]))]
                
                df = pd.DataFrame(data_array, columns=columns)
                return df, query_text
    
    # If no attachments or no data in attachments, return text content
    if 'content' in complete_message:
        return complete_message.get('content', ''), None
    
    return "No response available", None

# --- REFACTORED MAIN FUNCTION (Request 1) ---
def genie_query(question: str, conversation_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Union[str, pd.DataFrame], Optional[str]]:
    """
    Main entry point for querying Genie.
    Manages session context by deciding to start or continue a conversation.
    
    Args:
        question: The question to ask
        conversation_id: The existing conversation ID, if any.
        
    Returns:
        Tuple containing:
        - conversation_id: The ID for this conversation (new or existing)
        - message_id: The ID for the bot's response message
        - result: Either text_response or dataframe
        - query_text: The SQL query, if any
    """
    try:
        if conversation_id is None:
            logger.info("Starting new conversation...")
            # This is a new conversation
            conv_id, msg_id, result, query_text = start_new_conversation(question)
            
            # Check if the conversation failed to start (e.g., error)
            if conv_id is None:
                 return None, None, result, None
                 
            return conv_id, msg_id, result, query_text
        else:
            logger.info(f"Continuing conversation {conversation_id}...")
            # This is an existing conversation
            msg_id, result, query_text = continue_conversation(conversation_id, question)
            
            # Handle the specific case where the conversation expired
            if msg_id is None and "expired" in str(result):
                logger.warning(f"Conversation {conversation_id} expired. Clearing session.")
                # Return None for conv_id to signal the UI to clear its state
                return None, None, result, None
                
            return conversation_id, msg_id, result, query_text
            
    except Exception as e:
        logger.error(f"Error in genie_query: {str(e)}. Please try again.")
        return conversation_id, None, f"Sorry, an error occurred: {str(e)}. Please try again.", None

# --- NEW HELPER FUNCTION (Request 2) ---
def record_feedback(conversation_id: str, message_id: str, feedback_type: str) -> Tuple[bool, str]:
    """
    High-level helper to record feedback for a specific message.
    
    Args:
        conversation_id: The conversation ID
        message_id: The message ID that received feedback
        feedback_type: The feedback given (e.g., "POSITIVE" or "NEGATIVE")
        
    Returns:
        Tuple (success_boolean, status_message)
    """
    if not all([conversation_id, message_id, feedback_type]):
        logger.warning("Feedback attempted with missing IDs.")
        return False, "Missing conversation_id or message_id"
        
    logger.info(f"Recording feedback for message {message_id} in conversation {conversation_id}: {feedback_type}")
    
    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID
    )
    
    try:
        # Use UPPERCASE for consistency, as an example
        response = client.send_feedback(conversation_id, message_id, feedback_type.upper())
        logger.info(f"Feedback API response: {response}")
        return True, "Feedback recorded successfully"
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        return False, f"Error recording feedback: {str(e)}"
