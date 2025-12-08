import React, { useState } from 'react';
import { runGenieQuery } from './genieApi';

// Constants
const DATABRICKS_HOST = "adb-xxxx.xx.azuredatabricks.net";
const SPACE_ID = "01ef...";
const MY_TOKEN = "dapi..."; 

const GenieApp = () => {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  
  // --- STATE TO REMEMBER CONVERSATION ID ---
  const [conversationId, setConversationId] = useState<string | null>(null);
  
  // State for display
  const [displayResult, setDisplayResult] = useState<any>(null);

  const handleSearch = async () => {
    if(!input) return;
    setLoading(true);

    // Pass the 'conversationId' state into the function
    const response = await runGenieQuery(
        input, 
        DATABRICKS_HOST, 
        SPACE_ID, 
        MY_TOKEN, 
        conversationId // <--- PASS IT HERE
    );

    setLoading(false);

    if (typeof response === 'string') {
        // Handle Error
        alert(response);
    } else {
        // Success!
        // 1. Save the conversation ID for the NEXT turn
        setConversationId(response.conversationId);
        
        // 2. Show the result
        setDisplayResult(response);
    }
  };

  const clearChat = () => {
      setConversationId(null); // Reset ID to force a new conversation next time
      setDisplayResult(null);
      setInput("");
  }

  return (
    <div style={{ padding: 20 }}>
      <h3>Databricks Genie</h3>
      
      <div style={{marginBottom: 10}}>
         Status: {conversationId ? "Continuing Conversation..." : "New Conversation"}
         <button onClick={clearChat} style={{marginLeft: 10, fontSize: '0.8rem'}}>Reset Chat</button>
      </div>

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        placeholder="Ask a question..."
      />
      <button onClick={handleSearch} disabled={loading}>
        {loading ? "Thinking..." : "Send"}
      </button>

      {/* Render Results (Simplified) */}
      <div style={{ marginTop: 20 }}>
        {displayResult?.text && <p>{displayResult.text}</p>}
        
        {displayResult?.rows && (
            <table border={1}>
                <thead>
                    <tr>{displayResult.columns.map((c:string) => <th key={c}>{c}</th>)}</tr>
                </thead>
                <tbody>
                    {displayResult.rows.map((row:any[], i:number) => (
                        <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
                    ))}
                </tbody>
            </table>
        )}
      </div>
    </div>
  );
};

export default GenieApp;
