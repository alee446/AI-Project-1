from importlib.metadata import metadata
import trafilatura
import requests
import google.generativeai as genai
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# ---------------------- API CONFIGURATION ----------------------
import streamlit as st

TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


# TODO: NEWS SEARCH AND SUMMARY PREPARATION FUNCTIONS:

# ---------------------- TAVILY SEARCH ----------------------

def get_tavily_results(topic):
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    data = {
        "api_key": TAVILY_API_KEY,
        "query": topic,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        search_data = response.json().get("results", [])
        return [
            {
                "title": item["title"],
                "url": item["url"],
                "snippet": item["content"]
            } for item in search_data
        ]
    except Exception as e:
        print("Error from Tavily:", e)
        return []

# ---------------------- SERPER SEARCH ----------------------

def get_serper_results(topic):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    data = {"q": topic}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        search_data = response.json().get("organic", [])
        return [
            {
                "title": item["title"],
                "url": item["link"],
                "snippet": item.get("snippet", "")
            } for item in search_data
        ]
    except Exception as e:
        print("Error from Serper:", e)
        return []

# ---------------------- GEMINI SUMMARY ----------------------

def summarize_with_gemini(results, topic):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    combined_text = ""
    for result in results:
        combined_text += f"- {result['title']}\n  {result['snippet']}\n  🔗 {result['url']}\n\n"

    prompt = f"""
    You are an expert AI summarizer. Below are search results about the topic: "{topic}".

    Write a structured summary with:
    - 5 numbered **headings** (one for each key point).
    - Under each heading, include a **short paragraph** (2–3 sentences) explaining it clearly.
    - At the end, add a "**Sources**" section that lists all the original URLs. MUST PROVIDE THE SOURCE!! YOUR ANSWER IS INCOMPLETE WITHOUT THE URLS
    NOTE: FOLLOW THE PATTERN MENTIONED. IT IS MUST!!!
    Use friendly and informative language. Do not repeat the same link twice.

    Here is the search information:

    {combined_text}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# TODO: EMBEDDING, UPSERTING AND RETRIEVING FUNCTIONS:

# ---------------------- EMBEDDIG NEWS SUMMARY FUNCTION ----------------------

def embed_news_article_summary(article_data):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                   google_api_key=GEMINI_API_KEY)

    documents = []
    texts = []

    for item in article_data:
        content = f"{item['title']} \n\n {item['snippet']}"
        texts.append(content)

        documents.append(Document(
            page_content= content,
            metadata = {"title": item["title"],
                        "url": item["url"]}
        ))

    embeddings = embedding_model.embed_documents(texts)
    return list(zip(documents, embeddings))

# ---------------------- UPSERTING NEWS SUMMARY FUNCTION ----------------------

def upsert_news_article_summary(embeddings_with_documents):
    pc = Pinecone(api_key = PINECONE_API_KEY, environment = "gcp-starter")

    index_name = "news-summary"
    dimension = len(embeddings_with_documents[0][1])
    metric = "cosine"

    if index_name in [index.name for index in pc.list_indexes()]:
        pc.delete_index(index_name)

    pc.create_index(
        name = index_name,
        dimension = dimension,
        metric = metric,
        spec = ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
        )
    )

    index = pc.Index(index_name)

    vectors_to_upsert = [
        {
            "id": f"article_{i}",
            "values": embedding,
            "metadata": {
                "title": doc.metadata["title"],
                "url": doc.metadata["url"],
                "text": doc.page_content
            }
        }
        for i, (doc, embedding) in enumerate(embeddings_with_documents)
    ]
    index.upsert(vectors=vectors_to_upsert)

# ---------------------- RETRIEVING NEWS SUMMARY FUNCTION ----------------------

def retrieve_news_article_summary(query):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    embedded_query = embedding_model.embed_documents([query])[0]

    # Connect to Pinecone and query
    pc = Pinecone(api_key=PINECONE_API_KEY, environment="gcp-starter")
    index = pc.Index("news-summary")

    query_result = index.query(
        vector=[embedded_query],
        top_k=5,
        include_metadata=True
    )

    text = [match['metadata']['text'] for match in query_result["matches"]]
    url = [match['metadata']['url'] for match in query_result["matches"]]
    return text, url


# TODO: FUNCTION FOR RAG OF SUMMARY:

# ---------------------- RAG NEWS SUMMARY FUNCTION ----------------------

def rag_news_summary(query: str, message_log: list = None) -> str:
    context = retrieve_news_article_summary(query=query)

    # Fetch recent conversation history (last 4 exchanges)
    history_str = ""
    if message_log:
        history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in message_log[-4:]])

    prompt = f"""
You are an intelligent assistant. Use the following context and conversation history to generate a clear, accurate, and helpful answer to the user’s query. 
Only use the given context—do not make up information.

### Conversation History:
{history_str}

### Context:
{context}

### User Query:
{query}

### Your Answer: 
"""

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"


# TODO: FINAL TOOL FUNCTIONS:

def summarize_store_news(topic: str, max_results: int = 7) -> str:

    # Step 1: Find and summarize news
    print(f"🔍 Searching for: {topic}")
    tavily_news = get_tavily_results(topic)
    serper_news = get_serper_results(topic)
    all_news = tavily_news + serper_news
    selected_news = all_news[:max_results]

    if not selected_news:
        return "❌ No relevant news found."

    summary = summarize_with_gemini(selected_news, topic)

    # Step 2: Store data in Pinecone
    embeddings_with_documents = embed_news_article_summary(selected_news)
    upsert_news_article_summary(embeddings_with_documents)

    # Step 3: Return summary only
    return summary

import json

# def rag_news(query: str):
#     raw = rag_news_summary(query=query)
#
#     # Extract key fields manually
#     msg_match = re.search(r"\[message\]: (.+)", raw)
#     url_match = re.search(r"\[url\]: (.+)", raw)
#     response_match = re.search(r"\[response\]: (.+)", raw, re.DOTALL)
#
#     if not msg_match:
#         return "❌ Failed to extract [message] from LLM response."
#
#     message = msg_match.group(1).strip()
#
#     if message == "not-enough-context":
#         url = url_match.group(1).strip() if url_match else None
#         if not url:
#             return "❌ URL missing in not-enough-context response."
#         full_answer = rag_news_article(url=url, query=query)
#         return {
#             "msg": "✅ Answer retrieved from full article (due to insufficient summary context).",
#             "answer": full_answer
#         }
#
#     elif message == "answer":
#         answer = response_match.group(1).strip() if response_match else "❌ Response missing."
#         return {
#             "msg": "✅ Answer retrieved from Pinecone summary (no need to fetch article).",
#             "answer": answer
#         }
#
#     return "❌ Unknown [message] value in response."



#print(rag_news_article(url = "https://www.bbc.com/news/articles/c20rrxjnx4lo", query = "What are the major ideological divisions within the Republican Party regarding Trump's mega-bill, and how do these internal conflicts reflect broader challenges in passing large-scale legislation in a polarized Congress?"))
# response = rag_news(query= """According to Anthropic’s Dario Amodei, what is the long-term vision for AI in the field of biology, and how does this relate to the idea of a 'virtual biologist' mentioned in his 2024 manifesto?""")
# print(response)
#print(summarize_store_news(topic= "what are some new developments in AI Agents"))

#print(data_retriever("""According to Anthropic’s Dario Amodei, what is the long-term vision for AI in the field of biology, and how does this relate to the idea of a 'virtual biologist' mentioned in his 2024 manifesto?"""))
