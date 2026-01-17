import os
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from .database import similarity_search

# Ensure keys exist before initializing to avoid 500 crashes
GROQ_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=GROQ_KEY
)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

search_tool = Tool(
    name="search_documents",
    func=similarity_search,
    description="Use this tool to search the uploaded documents for specific facts."
)

# This is the most robust way to initialize an agent on Vercel
agent_executor = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)