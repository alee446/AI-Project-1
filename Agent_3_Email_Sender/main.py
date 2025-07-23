import sys
from pathlib import Path

from Agent_1_File_Assistant.main import GEMINI_API_KEY

sys.path.append(str(Path(__file__).parent))

# -------------------- Importing Necessary Pckages ---------------------
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import create_tool_calling_agent, AgentExecutor

from Agent_3_Email_Sender.email_generator_func import generate_email
from Agent_3_Email_Sender.email_editor_func import editor
from Agent_3_Email_Sender.email_send_func import send_email
from Agent_3_Email_Sender.extract_email_functions import extract_email_data
import streamlit as st
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# ----------------------- LLM SETUP -----------------------

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=500,
    google_api_key=GEMINI_API_KEY
)

# ----------------------- MEMORY -----------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY
)

chat_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=600,
    memory_key="message_log",
    return_messages=True
)

# ------------------------- Chat Template ----------------------------
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant with access to multiple tools.
            1. Email Generator: Use this tool to generate fresh email. You Must Access this tool When the User
            explicitly asks you to generate a FRESH EMAIL.
            ⚠️ When using this tool, always pass the user's **original prompt or instruction**, not a rewritten or drafted email. 
            2. Email Editor: Use this to make changes in the previously generated email. Use when the user wants 
            changes for the parts of the email like increasing or decreasing length, changing any specific content so on. 
            don't use this when the user wants to generate completely new email; use Email Generator for that 
            3. Email Sender: use this when the user is ok with the email. this tool sends the email so be careful. Use 
            when user says something like: I liked the email SEND IT or send the email I am ok with that. Must Confirm before 
            you send the email. 
            Be careful and pick the right tool based on the user’s actual intent. Your accuracy depends on correct tool choice."""),
    MessagesPlaceholder(variable_name="message_log"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# ------------------------- Tool 1: Generator ------------------------
email_generator_tool = Tool(
    name= "EmailGenerator",
    func= generate_email,
    description= """This is a tool that Generates Fresh Email content. 
    use this tool when the user asks for a new email to be generated. 
    DON'T use this when the user wants to make small changes on existing generated email.
    This tool shall be used when explicitly mentioned to Generate New or Fresh Email"""
)

# ------------------------ Tool 2: Editor ----------------------
email_editor_tool = Tool(
    name = "EmailEditor",
    func = editor,
    description = """This tool Edits on an existing generated email.
    Use this when the user wants to make changes on an email like the length, specific content etc.
    Don't use this when the user wants a NEW email rather than edited one"""
)

# ---------------------- Tool 3: Sender ------------------------
email_sender_tool = Tool(
    name = "EmailSender",
    func = send_email,
    description = """This tool sends the final generated email.
    use this when the user explicitly states that they are OK with the email and they want you TO SEND it.
    Don't use this if the user is still not convinced with the email and don't ask you to send it."""
)



# ----------------------- CREATE TOOL-CALLING AGENT -----------------------

tools = [email_generator_tool, email_editor_tool, email_sender_tool]

agent = create_tool_calling_agent(
    llm=chat,
    tools=tools,
    prompt=chat_prompt_template
)

# ----------------------- EXECUTOR -----------------------

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=chat_memory,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)


# while True:
#     user_query = input("YOU: ")
#     response = agent_executor.invoke({"input": user_query})
#     print(response["output"])
#

