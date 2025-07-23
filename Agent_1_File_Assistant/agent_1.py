# file_agent_app.py

import streamlit as st
import os
from main import agent_executor

from functions import data_loader_extracter
from functions import data_embedder
from functions import data_upserter

def run_file_agent():
    st.set_page_config(page_title="AI File Agent", layout="centered")
    st.title("📄 AI Agent: File-Based Q&A")

    # -------------------- SESSION STATE --------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False

    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""

    # -------------------- FILE UPLOAD SECTION --------------------
    uploaded_file = st.file_uploader("Upload a file to begin", type=["txt", "pdf", "md"])

    if uploaded_file and uploaded_file.name != st.session_state.uploaded_filename:
        # Save uploaded file
        file_path = os.path.join("temp_files", uploaded_file.name)
        os.makedirs("temp_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Step 1: Load
        with st.spinner("Loading file..."):
            loaded_data = data_loader_extracter(file_path)
            st.success("✅ File loaded successfully.")

        # Step 2: Embed
        with st.spinner("Generating embeddings..."):
            embedded_data = data_embedder(loaded_data)
            st.success("✅ Embeddings created successfully.")

        # Step 3: Upsert
        with st.spinner("Upserting into vector store..."):
            upsert_status = data_upserter(embedded_data)
            st.success("✅ Data upserted successfully into vector store.")

        # Update session state
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.file_processed = True
        st.session_state.chat_history = []

        st.toast("🎉 File setup complete! You can now start chatting with the AI.", icon="🤖")

    # -------------------- CHAT INTERFACE --------------------
    if st.session_state.file_processed:
        st.subheader("💬 Chat with your AI Agent")
        st.caption(f"Current file: `{st.session_state.uploaded_filename}`")

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"🧑‍💻 **You:** {msg['content']}")
            else:
                st.markdown(f"🤖 **AI:** {msg['content']}")

        # Chat input
        user_input = st.chat_input("Type your question here...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("🤖 Thinking..."):
                response = agent_executor.invoke({"input": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
                st.rerun()
    else:
        st.info("Please upload a file to begin chatting with the AI.")
