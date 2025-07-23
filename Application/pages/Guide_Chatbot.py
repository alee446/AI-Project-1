# guide_chatbot_app.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from config import GEMINI_API_KEY  # Make sure you store your key here safely

# ------------------- Setup Gemini --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5
)

# ------------------- System Prompt --------------------
system_prompt = """
You are the Guide Chatbot for Alveo AI. Help users understand and use its three main agents:

1. Nexa AI: Upload PDF, ask questions (RAG), summarize, and web search. It just accepts PDFs. 
2. NewsFlow AI: Enter a topic, fetch 5 articles, summarize, ask questions, web search.
3. AutoMail AI: Generate, edit, send emails. Users need to provide Recepient's Name, Email, Important details to include in the Email and the tone of email: friendly or business.
this agent handles everything from generation, editing to sending. The user just needs to command properly

THE DEVELOPER: 
Alveo AI is a multi agent application developed by a self taught Agentic AI Engineer, Ali Seena Ghulami on July 20, 2025.
Your role is to guide users, answer questions like “How do I use Nexa?”, and explain features clearly. Don’t perform tasks yourself — only guide users.
"""

# ------------------- App Interface --------------------
def run_guide_chatbot():
    st.set_page_config(page_title="Alveo Guide", layout="centered")
    st.title("ALVEO GUIDE CHATBOT")

    if "guide_chat_history" not in st.session_state:
        st.session_state.guide_chat_history = []

    st.caption("Ask me anything about how to use Alveo AI.")
    st.divider()

    # Display previous chat
    for msg in st.session_state.guide_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about Nexa, NewsFlow, or the Email Agent...")

    if user_input:
        st.session_state.guide_chat_history.append({"role": "user", "content": user_input})

        # Display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_messages = [
                    SystemMessage(content=system_prompt),
                    *[HumanMessage(content=msg["content"]) for msg in st.session_state.guide_chat_history if msg["role"] == "user"],
                ]
                response = llm.invoke(chat_messages)
                st.markdown(response.content)
                st.session_state.guide_chat_history.append({"role": "assistant", "content": response.content})
                st.rerun()

if __name__ == "__main__":
    run_guide_chatbot()
