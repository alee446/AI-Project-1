
from config import GEMINI_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from pinecone import Pinecone
import os

# ----------------------- INITIALIZING THE API KEYS -----------------------
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
api_key_gemini = GEMINI_API_KEY
api_key_pinecone = PINECONE_API_KEY

# ----------------------- RETRIEVE CONTEXT FROM VECTOR DB -----------------------

def run_semantic_search(query):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key_gemini
    )
    embedded_query = embedding_model.embed_documents([query])[0]
    pc = Pinecone(api_key=api_key_pinecone, environment="gcp-starter")
    index = pc.Index("ai-research-assistant")
    query_result = index.query(
        vector=[embedded_query],
        top_k=5,
        include_metadata=True
    )
    context = [match['metadata']['text'] for match in query_result["matches"]]
    return "\n".join(context)

# ----------------------- GET ENTIRE CONTEXT FOR SUMMARIZATION -----------------------

def run_full_context_retrieval():
    pc = Pinecone(api_key=api_key_pinecone, environment="gcp-starter")
    index = pc.Index("ai-research-assistant")
    query_result = index.query(
        vector=[[0.0] * 768],
        top_k=100,
        include_metadata=True
    )
    context_chunks = [match['metadata']['text'] for match in query_result["matches"]]
    return "\n\n".join(context_chunks)

# ----------------------- LLM SETUP -----------------------

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=1500,
    google_api_key=api_key_gemini
)

# ----------------------- TOOL 1: FileQA -----------------------

def file_qa_tool_func(question: str, message_log: list = None) -> str:
    context = run_semantic_search(question)
    message_log = chat_memory.chat_memory.messages

    history_str = ""
    if message_log:
        history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in message_log[-4:]])

    prompt = f"""
You will receive a question from a student about a file they have uploaded.
Only use the provided file context to answer.
Here is the previous conversation:

{history_str}

Question:
{question}

Context from file:
{context}
"""
    response = chat.invoke(prompt)
    return response.content

file_qa_tool = Tool(
    name="FileQA",
    func=file_qa_tool_func,
    description=(
        "Use this tool when the user asks a question related to the uploaded file. "
        "This tool performs a semantic search in the document's content and answers based on it."
    )
)

# ----------------------- TOOL 2: FileSummarizer -----------------------

def summarize_file_tool_func(_: str, message_log: list = None) -> str:
    full_context = run_full_context_retrieval()

    message_log = chat_memory.chat_memory.messages

    history_str = ""
    if message_log:
        history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in message_log[-4:]])

    prompt = f"""
You are an assistant that summarizes technical and educational documents.
Use the context below to summarize the file. If the user mentioned a specific goal earlier, consider it.

Conversation history:
{history_str}

Document content:
{full_context}
"""
    response = chat.invoke(prompt)
    return response.content

file_summarizer_tool = Tool(
    name="FileSummarizer",
    func=summarize_file_tool_func,
    description=(
        "Use this tool when the user asks to summarize the uploaded file. "
        "It generates a clear summary of the document, optionally considering user goals."
    )
)


# ----------------------- TOOL 3: WebSearch -----------------------

tavily_tool = TavilySearchResults()

def clean_web_search_tool_func(query: str, message_log: list = None) -> str:
    raw_results = tavily_tool.run(query)

    top_contents = ""
    if isinstance(raw_results, list):
        top_contents = "\n\n".join([r.get("content", "") for r in raw_results[:3]])

    message_log = chat_memory.chat_memory.messages

    history_str = ""
    if message_log:
        history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in message_log[-4:]])

    prompt = f"""
You are an AI assistant summarizing online search results for the user.

User's recent conversation:
{history_str}

Here are the search results:
{top_contents}

Please write a short, clear, helpful response based on this.
"""
    response = chat.invoke(prompt)
    return response.content

web_search_tool = Tool(
    name="WebSearch",
    func=clean_web_search_tool_func,
    description=(
        "Use this tool only when the user explicitly asks to 'SEARCH ONLINE'. "
        "Summarizes real-time search results, optionally considering recent conversation context."
    )
)

# ----------------------- MEMORY -----------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key_gemini
)

chat_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=600,
    memory_key="message_log",
    return_messages=True
)

# ----------------------- CUSTOM PROMPT TEMPLATE -----------------------

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant with access to multiple tools.
            Only use the WebSearch tool if the user asks for information from the internet or real-time data. 
            Examples include current events, dates, live news, or explicit mentions of "search online" or "find on the web."
            If the user asks a question that seems related to a course, document, uploaded file, or educational content, 
            always prefer the FileQA tool.
            If uncertain, assume the user is referring to the uploaded file unless they clearly ask for something online.
            Be careful and pick the right tool based on the user’s actual intent. Your accuracy depends on correct tool choice."""),
    MessagesPlaceholder(variable_name="message_log"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ----------------------- CREATE TOOL-CALLING AGENT -----------------------

tools = [file_qa_tool, file_summarizer_tool, web_search_tool]

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
    return_intermediate_steps=False
)

response = agent_executor.invoke({"input": "tell me a very common myths people have about teachers. use the file I have uploaded"})
print(response["output"])
