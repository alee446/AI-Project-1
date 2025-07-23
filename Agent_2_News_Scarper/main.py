from .functions import summarize_store_news, rag_news_summary
from config import GEMINI_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_tavily import TavilySearch
from pinecone import Pinecone
import os

# ----------------------- INITIALIZING THE API KEYS -----------------------
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
api_key_gemini = GEMINI_API_KEY
api_key_pinecone = PINECONE_API_KEY

# ----------------------- LLM SETUP -----------------------

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Most stable free-tier model
    temperature=0,
    max_output_tokens=2048,  # Increased for better responses
    google_api_key=api_key_gemini,
    convert_system_message_to_human=True  # Helps with prompt formatting
)

# ----------------------- MEMORY -----------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Consistent with main LLM
    google_api_key=api_key_gemini,
    convert_system_message_to_human=True  # Match main LLM settings
)

chat_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=600,
    memory_key="chat_history",  # Changed from "message_log"
    return_messages=True
)
# ----------------------- TOOL 1: News Scraper -----------------------

news_scraper = Tool(
    name="News-Scraper",
    func=summarize_store_news,
    description=(
        "Use this tool **ONLY when the user explicitly asks for a 'news summary', 'latest news', "
        "or to 'scrape news' about a specific topic.** "
        "It actively searches for and fetches *new, recent* news articles related to the query, "
        "summarizes their content, and returns both the summary and the article URLs. "
        "Do not use this for general questions or to answer from previously acquired knowledge."
    )
)

# ----------------------- TOOL 2: RAG of News Summary -----------------------

news_qa_tool = Tool(
    name="NewsQA",
    func=rag_news_summary,
    description=(
        "Use this tool to **answer questions about news topics ONLY from the previously stored news data.** "
        "It does not search the internet or scrape new articles. "
        "It leverages saved article summaries, titles, and URLs from past news scraping "
        "to provide comprehensive answers to user queries based on existing knowledge."
    )
)

# ----------------------- TOOL 3: WebSearch -----------------------

tavily_tool = TavilySearch(
    max_results=5,
    topic="general"
)

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
        "Use this tool **ONLY when the user explicitly asks to 'search online', 'look something up on the internet', "
        "or for 'real-time information' on a general topic (not specifically news).** "
        "It performs a real-time web search and summarizes the results. "
        "Do not use this for news summaries (use News-Scraper for that) or for questions that can be answered from stored news (use NewsQA for that)."
        "you must use this if no results are obtained from NewsScraper and NewsQA"
    )
)

# ----------------------- CUSTOM PROMPT TEMPLATE -----------------------

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
            "You are a helpful news assistant with access to three tools: NewsQA, NewsScraper, and WebSearch.
            Use the **WebSearch** tool only when the user explicitly asks to 'search online', mentions the 'internet', or requests real-time information such as live updates, current events, dates, or breaking news.
            You must use web search if YOU COULDN'T GET RESULTS FROM NewsScraper or NewsQA.
            Use the **NewsScraper** tool when the user wants a summary of the latest news articles on a specific topic. This tool fetches and summarizes fresh news from the web.
            Use the **NewsQA** tool when the user asks questions related to previously processed news topics. It answers using stored summaries, titles, and URLs — it does not fetch new data.
            when user asks for any live news, use NewsScraper if couldn't retrieve answer use Websearch but give an output. 
            Select the most appropriate tool based on the user's exact request. Your effectiveness depends on choosing the right tool for the right context."""),
    MessagesPlaceholder(variable_name="chat_history"),  # Changed from "message_log"
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ----------------------- CREATE TOOL-CALLING AGENT -----------------------

tools = [news_scraper, news_qa_tool, web_search_tool]

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

def ask_agent_1(query):
    response = agent_executor.invoke({"input": query}, memory = chat_memory)
    print(response["output"])