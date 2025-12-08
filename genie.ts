// genieApi.ts

// --- Types ---
interface GenieMessage {
  id: string;
  status: 'COMPLETED' | 'EXECUTING' | 'failed' | 'error';
  attachments?: Array<{
    attachment_id: string;
    query?: { query: string }; 
    text?: { content: string };
  }>;
  content?: string;
}

interface GenieResult {
  conversationId: string; // We return this so React can save it
  messageId: string;
  sql?: string;
  columns?: string[];
  rows?: any[][];
  text?: string;
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const runGenieQuery = async (
  question: string,
  host: string,
  spaceId: string,
  token: string,
  existingConversationId?: string | null // <--- NEW PARAMETER
): Promise<GenieResult | string> => {
  
  const baseUrl = `https://${host}/api/2.0/genie/spaces/${spaceId}`;
  const headers = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };

  try {
    let conversationId = existingConversationId;
    let messageId = "";

    // --- STEP 1: DECIDE ENDPOINT (Start vs Continue) ---
    if (conversationId) {
      // SCENARIO A: Continue Existing Conversation
      console.log(`Continuing conversation: ${conversationId}`);
      const res = await fetch(`${baseUrl}/conversations/${conversationId}/messages`, {
        method: "POST",
        headers,
        body: JSON.stringify({ content: question }),
      });
      if (!res.ok) throw new Error(`Continue failed: ${res.statusText}`);
      const data = await res.json();
      messageId = data.message_id; // API returns message_id
    } else {
      // SCENARIO B: Start New Conversation
      console.log("Starting NEW conversation...");
      const res = await fetch(`${baseUrl}/start-conversation`, {
        method: "POST",
        headers,
        body: JSON.stringify({ content: question }),
      });
      if (!res.ok) throw new Error(`Start failed: ${res.statusText}`);
      const data = await res.json();
      conversationId = data.conversation_id; // API returns new conversation_id
      messageId = data.message_id;
    }

    // --- STEP 2: POLL UNTIL COMPLETE (Same as before) ---
    let status = "EXECUTING";
    let messageData: GenieMessage | null = null;

    while (status === "EXECUTING") {
      await delay(2000); 
      const msgRes = await fetch(
        `${baseUrl}/conversations/${conversationId}/messages/${messageId}`,
        { headers }
      );
      messageData = await msgRes.json();
      status = messageData?.status || "failed";
    }

    if (status !== "COMPLETED" || !messageData) {
      return "Error: Genie failed to generate a response.";
    }

    // --- STEP 3: EXTRACT DATA ---
    // Prepare the base return object
    const finalResult: GenieResult = {
        conversationId: conversationId!, // ! asserts it's not null now
        messageId: messageId
    };

    const queryAttachment = messageData.attachments?.find((a) => a.query);

    if (queryAttachment) {
      // It's a Table/SQL result
      finalResult.sql = queryAttachment.query?.query;

      const resultRes = await fetch(
        `${baseUrl}/conversations/${conversationId}/messages/${messageId}/attachments/${queryAttachment.attachment_id}/query-result`,
        { headers }
      );
      const resultJson = await resultRes.json();
      const statement = resultJson.statement_response;

      finalResult.columns = statement.manifest.schema.columns.map((c: any) => c.name);
      finalResult.rows = statement.result.data_array;
    } else {
      // It's a Text result
      finalResult.text = messageData.content;
    }

    return finalResult;

  } catch (error) {
    console.error(error);
    return "API Request Failed";
  }
};
