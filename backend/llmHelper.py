from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import Tool
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.tools import DuckDuckGoSearchRun
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.agents import initialize_agent, AgentType
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
import os
from data import emails

def email_search(
    email_content: Optional[str] = None,
    sender: Optional[str] = None,
    timestamp: Optional[str] = None,
    similarity_threshold: float = 0.25
) -> List[dict]:
    if not any([email_content, sender, timestamp]):
        raise ValueError("At least one parameter (email_content, sender, or timestamp) must be provided.")
    
    results = emails  
    
    if sender:
        sender = sender.strip().lower()
        results = [email for email in emails if email['sender'].strip().lower() == sender]
    
    if timestamp:
        try:
            target_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            results = [
                email for email in results
                if datetime.strptime(email['timestamp'], "%Y-%m-%d %H:%M:%S") == target_timestamp
            ]
        except ValueError:
            raise ValueError("Timestamp must be in the format 'YYYY-MM-DD HH:MM:SS'.")
    
    if email_content and results:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(email_content, convert_to_numpy=True)
        email_embeddings = model.encode([email['content'] for email in results], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding.reshape(1, -1), email_embeddings).flatten()
        results = [email for email, sim in zip(results, similarities) if sim >= similarity_threshold]
    
    return results

def web_search(query: str) -> str:
    search = DuckDuckGoSearchRun()
    return search.run(query)

def company_website_search(query: str) -> str:
    urls = ["https://carnotresearch.com/"]  
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=['\n\n', '\n', '.', ',']
    )
    
    docs = text_splitter.split_documents(data)
    model = "sentence-transformers/all-mpnet-base-v2"
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    embeddings = HuggingFaceHubEmbeddings(
        model=model,
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )
    
    vectorindices = FAISS.from_documents(docs, embeddings)
    similar_docs = vectorindices.similarity_search(query, k=3)
    
    return "\n\n".join([doc.page_content for doc in similar_docs])

def getLlmResponse(query):
    load_dotenv()
    
    email_search_tool = Tool.from_function(
        func=email_search,
        name="email_search",
        description="Searches through emails to locate messages based on content, sender, or timestamp."
    )
    
    web_search_tool = Tool.from_function(
        func=web_search,
        name="web_search",
        description="Executes a search query on the internet to retrieve relevant web pages, articles, or resources."
    )
    
    company_website_search_tool = Tool.from_function(
        func=company_website_search,
        name="company_website_search",
        description="Searches the company website for relevant information based on the query."
    )
    
    system_prompt = """
        You are an intelligent assistant that helps users find emails, search the web, and search the company website.
        Use the following guidelines:
        1. First, analyze the user's query and break it down into its components.
        2. Explain which tool will be used and why.
        3. Use the appropriate tool to retrieve the information.
        4. Provide a clear and concise response, including the breakdown of the query.
    """
    
    few_shot_examples = [
        HumanMessage(content="Find emails from ramesh@company.com"),
        SystemMessage(content='{"action": "email_search", "action_input": {"sender": "ramesh@company.com"}}'),
        HumanMessage(content="Find the emails related to pull request"),
        SystemMessage(content='{"action": "email_search", "action_input": {"email_content": "pull request"}}'),
        HumanMessage(content="Find emails sent on 2025-02-21 10:05:14"),
        SystemMessage(content='{"action": "email_search", "action_input": {"timestamp": "2025-02-21 10:05:14"}}'),
        HumanMessage(content="Search the web for the latest news on AI"),
        SystemMessage(content='{"action": "web_search", "action_input": {"query": "latest news on AI"}}'),
        HumanMessage(content="Search the company website for information on the latest product release"),
        SystemMessage(content='{"action": "company_website_search", "action_input": {"query": "latest product release"}}'),
    ]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        system_message=SystemMessage(content=system_prompt),
        examples=few_shot_examples
    )
    
    tools = [email_search_tool, web_search_tool, company_website_search_tool]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    response = agent.run(
        f"""
        Analyze the following query and break it down into its components.
        Explain which tool will be used and why. Then, proceed with the search and include the breakdown in the final response.
        Query: {query}
        """
    )
    
    return response
