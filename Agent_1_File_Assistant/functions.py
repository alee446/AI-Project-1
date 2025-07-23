#todo: IMPORTING NECESSARY PACKAGES

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool

from langchain.agents import (create_tool_calling_agent,
                              AgentExecutor)

from langchain.memory import ConversationSummaryMemory

from langchain.text_splitter import TokenTextSplitter

from langchain_core.output_parsers import StrOutputParser

from config import GEMINI_API_KEY
from config import PINECONE_API_KEY

api_key_gemini = GEMINI_API_KEY
api_key_pinecone = PINECONE_API_KEY

import pinecone
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader

# todo: Loading and Extracting Data:

def data_loader_extracter(file):
    pdf_loader = PyPDFLoader(file)
    data = pdf_loader.load()

    data_splitter = TokenTextSplitter(encoding_name="cl100k_base",
                                      chunk_size=500,
                                      chunk_overlap=50)

    splitted_data = data_splitter.split_documents(data)

    return splitted_data

# todo: Embedding the Data

def data_embedder(data):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                             google_api_key=api_key_gemini)
    texts = [doc.page_content for doc in data]
    embeddings = embedding_model.embed_documents(texts)

    return list(zip(data, embeddings))

# todo: Upserting the Vectors to Pinecone

def data_upserter(embedded_data_with_text):
    pc = Pinecone(api_key=api_key_pinecone, environment="gcp-starter")

    index_name = "ai-research-assistant"
    dimension = len(embedded_data_with_text[0][1])
    metric = "cosine"

    # Delete index if it already exists
    if index_name in [index.name for index in pc.list_indexes()]:
        pc.delete_index(index_name)

    # Create new index
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    # Connect to the created index
    index = pc.Index(index_name)

    # Create list of (id, vector) tuples
    from langchain_core.documents import Document  # or langchain.schema if needed

    vectors_to_upsert = [
        {
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": doc.page_content
            }
        }
        for i, (doc, embedding) in enumerate(embedded_data_with_text)
    ]

    # Batch upsert
    batch_size = 1000
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)

# todo: Buidling the Agent

def data_retriever(query):
    # retrieving the vector
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                   google_api_key=api_key_gemini)
    embedded_query = embedding_model.embed_documents([query])[0]

    pc = Pinecone(api_key=api_key_pinecone, environment="gcp-starter")
    index = pc.Index("ai-research-assistant")
    query_result = index.query(
        vector = [embedded_query],
        top_k = 5,
        include_metadata=True
    )

    for match in query_result["matches"]:
            return match['metadata']['text']

