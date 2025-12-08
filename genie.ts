// src/genieService.ts

// --- CONFIGURATION ---
// TODO: Replace these with your actual keys or process.env variables
const DATABRICKS_HOST = "adb-YOUR_INSTANCE.azuredatabricks.net"; 
const SPACE_ID = "YOUR_SPACE_ID_HERE";
const TOKEN = "YOUR_PAT_TOKEN_HERE"; 

const BASE_URL = `https://${DATABRICKS_HOST}/api/2.0/genie/spaces/${SPACE_ID}`;

// --- TYPES ---
export interface GenieResult {
  conversationId: string;
  messageId: string;
  text: string | null;      // The conversational answer
  sql: string | null;       // The SQL query generated (if any)
  columns: string[] | null; // Table headers
  rows: any[][] | null;     // Table data
  error?: string;
}

// --- HELPER: Wait function for polling ---
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Main function to call Databricks Genie.
 * @param question - The user's input.
 * @param conversationId - (Optional) Pass the ID to continue a chat. Leave null to start new.
 */
export const sendMessageToGenie = async (
  question: string,
  conversationId: string | null
): Promise<GenieResult> => {

  const headers = {
    "Authorization": `Bearer ${TOKEN}`,
    "Content-Type": "application/json",
  };

  try {
    // ---------------------------------------------------------
    // STEP 1: Send Message (Start New OR Continue Existing)
    // ---------------------------------------------------------
    let endpoint = `${BASE_URL}/start-conversation`;
    let activeConversationId = conversationId;

    if (activeConversationId) {
      endpoint = `${BASE_URL}/conversations/${activeConversationId}/messages`;
    }

    const initRes = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({ content: question }),
    });

    if (!initRes.ok) {
      const errText = await initRes.text();
      throw new Error(`Init Failed: ${initRes.status} - ${errText}`);
    }

    const initData = await initRes.json();
    
    // If we started a new chat, the API gives us the new ID.
    if (!activeConversationId) {
      activeConversationId = initData.conversation_id;
    }
    const messageId = initData.message_id;

    // ---------------------------------------------------------
    // STEP 2: Poll for Completion
    // ---------------------------------------------------------
    let status = "EXECUTING";
    let messageData = null;

    while (status === "EXECUTING") {
      await delay(2000); // Wait 2 seconds between checks
      
      const pollRes = await fetch(`${BASE_URL}/conversations/${activeConversationId}/messages/${messageId}`, { 
        headers 
      });
      
      if (!pollRes.ok) throw new Error("Polling failed");
      
      messageData = await pollRes.json();
      status = messageData.status;

      if (status === "FAILED" || status === "ERROR") {
        return {
          conversationId: activeConversationId!,
          messageId,
          text: null, sql: null, columns: null, rows: null,
          error: "Genie encountered an error processing your request."
        };
      }
    }

    // ---------------------------------------------------------
    // STEP 3: Fetch Results (SQL & Data)
    // ---------------------------------------------------------
    // Check if there is an attachment with a SQL query
    const queryAttachment = messageData?.attachments?.find((a: any) => a.query);
    
    let sql = null;
    let columns = null;
    let rows = null;
    let text = messageData.content || null;

    if (queryAttachment) {
      sql = queryAttachment.query.query;
      
      // Fetch the actual data rows for this attachment
      const resultUrl = `${BASE_URL}/conversations/${activeConversationId}/messages/${messageId}/attachments/${queryAttachment.attachment_id}/query-result`;
      
      const dataRes = await fetch(resultUrl, { headers });
      const dataJson = await dataRes.json();
      
      const statement = dataJson.statement_response;
      // Extract columns and rows from Databricks format
      columns = statement.manifest.schema.columns.map((c: any) => c.name);
      rows = statement.result.data_array;
    }

    return {
      conversationId: activeConversationId!,
      messageId,
      text,
      sql,
      columns,
      rows
    };

  } catch (err: any) {
    console.error("Genie Service Error:", err);
    throw err; // Re-throw so UI can handle it
  }
};
