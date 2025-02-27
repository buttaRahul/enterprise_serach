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
import pymysql
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


def db_search(query: str) -> str:
    """
    Connects to the local MySQL database using pymysql and executes the provided SQL query.
    """
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="root",
            database="employeedb",
            cursorclass=pymysql.cursors.Cursor
        )
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Format results as a string
            result_str = "\n".join([str(row) for row in results])
            print(result_str)
            return result_str if result_str else "No results found."
    
    except pymysql.MySQLError as e:
        return f"Error: {e}"
    
    finally:
        connection.close()



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

    db_search_tool = Tool.from_function(
        func=db_search,
        name="db_search",
        description="Executes an SQL query on the local MySQL database and retrieves the results."
    )

    
    system_prompt = """
        You are an intelligent assistant that helps users find emails, search the web, search the company website, and query a database. 
        Always respond using the following structured format, ensuring **all steps are included**:

        ---
        ### **Response Format (Must Follow)**
        1. **Query Breakdown**: Identify key components of the query.
        2. **Function Used (Mandatory)**: Clearly specify which function is used (`email_search`, `web_search`, etc.).
        3. **Reasoning & Processing Steps**: Explain **why** this function is used and how the query is handled.
        4. **Final Search & Results**: Execute the function and return the output.

        **Example Response:**
        #### **Query Breakdown**
        - The user is looking for emails from `ramesh@company.com`.

        #### **Function Used (Mandatory)**
        - Using `email_search` because the request is about finding emails.

        #### **Reasoning & Processing Steps**
        - We search the email database using the sender filter.
        - Extract relevant emails based on the sender's address.

        #### **Final Search & Results**
        `{"action": "email_search", "action_input": {"sender": "ramesh@company.com"}}`

        ---

        ### **General Guidelines:**
        - If the query involves emails, use `email_search`.
        - If the query is about general knowledge, use `web_search`.
        - If it's about the company, use `company_website_search`.
        - If it contains an SQL statement, use `db_search`.
        - If multiple actions are needed, **explain each step explicitly**.

        **STRICT RULE**: **Always follow the 4-step response structure, or the response is invalid.**
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
        HumanMessage(content="Run this sql query select firstname,email from employee"),
        SystemMessage(content='{"action": "db_search", "action_input": {"query": "select firstname,email from employee"}}'),
        HumanMessage(content="Find emails related to bugs or issues and search online to find solutions for them"),
        SystemMessage(content='{"action": "multi_action", "action_input": [{"action": "email_search", "action_input": {"email_content": "bugs OR issues"}}, {"action": "web_search", "action_input": {"query": "<CONTENT_FROM_RETRIEVED_EMAILS>"}}]}'),

    ]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        system_message=SystemMessage(content=system_prompt),
        examples=few_shot_examples
    )
    
    tools = [email_search_tool, web_search_tool, company_website_search_tool, db_search_tool]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    response = agent.invoke(
        f"""
            **STRICT FORMAT REQUIRED** - Your final response **must** include:
            
            1. **Query Breakdown:** (Identify key components)
            2. **Function Used (Mandatory):** (List all functions used)
            3. **Reasoning & Processing Steps:** (Detailed breakdown)
            4. **Final Search & Results:** (Provide the final answer based on processed data)
            
            **Do not skip any step.** Even if the final result is computed, explicitly include all breakdown steps.
            
            Query: {query}
        """
    )
    print(response['output'])
    return response['output']
