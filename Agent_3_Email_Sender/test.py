import streamlit as st
from main import agent_executor  # replace 'main' with your actual file name if different

st.set_page_config(page_title="AI Email Assistant", layout="centered")

st.title("📧 AI Email Assistant")
st.markdown("Generate, edit, and send emails using an AI-powered agent.")

# Session state to store history and result
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = ""

# User input
user_input = st.text_area("Enter your request to the assistant:", height=150)

if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = agent_executor.invoke({"input": user_input})
            user_msg = user_input
            assistant_msg = result["output"]

            st.session_state.chat_history.append(("YOU", user_msg))
            st.session_state.chat_history.append(("Assistant", assistant_msg))
            st.session_state.last_result = assistant_msg  # store latest result to show clearly
    else:
        st.warning("Please enter a prompt to proceed.")

# Display latest tool result in focus
if st.session_state.last_result:
    st.subheader("📤 Latest Output")
    st.code(st.session_state.last_result, language="markdown")

# Display full chat history
st.subheader("💬 Full Conversation")
for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)

# Reset button
if st.button("🔄 Reset Conversation"):
    st.session_state.chat_history = []
    st.session_state.last_result = ""
    st.success("Conversation reset.")
