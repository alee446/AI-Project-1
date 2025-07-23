# email_agent_app.py

import streamlit as st
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Agent_3_Email_Sender.main import agent_executor


def run_email_agent():
    st.set_page_config(page_title="AutoMail", layout="centered")
    st.title("AUTO-MAIL AI")

    # ---------- SESSION STATE ----------
    if "email_chat_history" not in st.session_state:
        st.session_state.email_chat_history = []

    # ---------- INFO BOX ----------
    st.info("""
    ✉️ **How to Structure Your Prompt**

    To get the best results, please include:
    - **Recipient's Name** (e.g., Mr. John)
    - **Recipient's Email** (optional)
    - **Email Content or Purpose** (e.g., scheduling a meeting, sending a follow-up)
    - **Tone of the Email** (e.g., formal, friendly, apologetic)

    Example:  
    "Write a formal email to Mr. John at john@example.com to follow up on our last meeting about the project deadline."
    """)

    # ---------- CHAT INTERFACE ----------
    st.caption("Generate, edit, and send emails with the help of AI.")
    st.divider()

    # Display chat history
    for msg in st.session_state.email_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input bar
    user_input = st.chat_input("What would you like the email to say?")

    if user_input:
        st.session_state.email_chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = agent_executor.invoke({"input": user_input})
                st.markdown(response["output"])
                st.session_state.email_chat_history.append(
                    {"role": "assistant", "content": response["output"]}
                )
                st.rerun()


if __name__ == "__main__":
    run_email_agent()
