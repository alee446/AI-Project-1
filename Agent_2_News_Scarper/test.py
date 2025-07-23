# news_agent_app.py

import streamlit as st
from .main import agent_executor

def run_news_agent():
    st.set_page_config(page_title="News Insights Agent", layout="centered")
    st.title("AI Agent: News Analysis and Q&A")

    # ---------- SESSION STATE ----------
    if "news_chat_history" not in st.session_state:
        st.session_state.news_chat_history = []

    # ---------- CHAT INTERFACE ----------
    st.subheader("Chat with the News Agent")
    st.caption("Ask questions related to current events, summaries, or specific news topics.")

    # Display chat history
    for msg in st.session_state.news_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input bar
    user_input = st.chat_input("Ask something about the news...")

    if user_input:
        st.session_state.news_chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = agent_executor.invoke({"input": user_input})
                st.markdown(response["output"])
                st.session_state.news_chat_history.append(
                    {"role": "assistant", "content": response["output"]}
                )
                st.rerun()


if __name__ == "__main__":
    run_news_agent()
