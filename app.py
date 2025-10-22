import streamlit as st
import pandas as pd
import time
import random

# --- Page setup ---
st.set_page_config(page_title="💬 Smart Data Assistant", layout="wide")

# --- Custom CSS for beautification ---
st.markdown("""
<style>
.chat-container {
    background-color: #f9fafb;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.question-box {
    background-color: #e8f0fe;
    padding: 0.8rem;
    border-radius: 10px;
    font-weight: 600;
    color: #1a237e;
}
.answer-box {
    background-color: #ffffff;
    padding: 0.8rem;
    border-radius: 10px;
    color: #212121;
}
.feedback-btn {
    background-color: #f1f3f4;
    border-radius: 8px;
    padding: 0.3rem 0.6rem;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
}
.feedback-btn:hover {
    background-color: #e0e0e0;
}
.sql-box {
    background-color: #272822;
    color: #f8f8f2;
    padding: 0.5rem;
    border-radius: 8px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "show_sql" not in st.session_state:
    st.session_state.show_sql = {}

# --- Dummy backend ---
def generate_answer(question):
    """Simulated backend call returning a text answer, DataFrame, and SQL."""
    simulated_responses = [
        "Sure! Let’s pull up the data for that.",
        "Fetching the information you requested…",
        "Here’s what I found in the dataset.",
        "Let me show you a summary of that query."
    ]
    answer_text = random.choice(simulated_responses)
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Score": [89, 76, 92],
        "Passed": [True, False, True]
    })
    sql = f"SELECT * FROM students WHERE query = '{question}';"
    return answer_text, df, sql

# --- Input section ---
st.title("💬 Smart Data Assistant")

with st.container():
    cols = st.columns([8, 1])
    user_input = cols[0].text_input("Ask me anything:", placeholder="e.g., Show me top scoring students...")
    submit = cols[1].button("Send", use_container_width=True)

if submit and user_input.strip():
    with st.spinner("Thinking..."):
        # Insert empty response placeholder at top
        st.session_state.qa_history.insert(0, {
            "question": user_input,
            "answer_text": "",
            "answer_df": None,
            "sql": "",
            "feedback": None
        })

        # Simulate streaming effect
        answer_text, df, sql = generate_answer(user_input)
        partial_text = ""
        for ch in answer_text:
            partial_text += ch
            st.session_state.qa_history[0]["answer_text"] = partial_text
            time.sleep(0.03)
            st.experimental_rerun()

        # Once text done, add dataframe and SQL
        st.session_state.qa_history[0]["answer_df"] = df
        st.session_state.qa_history[0]["sql"] = sql
        if len(st.session_state.qa_history) > 50:
            st.session_state.qa_history = st.session_state.qa_history[:50]
        st.experimental_rerun()

# --- Display chat history ---
for i, qa in enumerate(st.session_state.qa_history):
    with st.container():
        st.markdown(f"<div class='chat-container'>", unsafe_allow_html=True)
        cols = st.columns(2)

        # Question
        with cols[0]:
            st.markdown(f"<div class='question-box'>🧠 {qa['question']}</div>", unsafe_allow_html=True)

        # Answer
        with cols[1]:
            st.markdown(f"<div class='answer-box'>{qa['answer_text']}</div>", unsafe_allow_html=True)

            if qa["answer_df"] is not None:
                st.dataframe(qa["answer_df"], use_container_width=True)
                if st.button(
                    "Show SQL" if not st.session_state.show_sql.get(i) else "Hide SQL",
                    key=f"sqlbtn_{i}",
                    use_container_width=True
                ):
                    st.session_state.show_sql[i] = not st.session_state.show_sql.get(i, False)
                if st.session_state.show_sql.get(i):
                    st.markdown(f"<div class='sql-box'>{qa['sql']}</div>", unsafe_allow_html=True)

                fcols = st.columns([1, 1, 1, 6])
                if fcols[0].button("👍", key=f"good_{i}"):
                    qa["feedback"] = "Good"
                if fcols[1].button("😐", key=f"neutral_{i}"):
                    qa["feedback"] = "Neutral"
                if fcols[2].button("👎", key=f"bad_{i}"):
                    qa["feedback"] = "Bad"
                if qa["feedback"]:
                    fcols[3].markdown(f"**Feedback:** {qa['feedback']}")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("✨ Powered by Streamlit — elegant, responsive, and interactive")
