import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from .database import similarity_search

def get_response(user_query, chat_history):
    llm = ChatGroq(
        temperature=0.1, 
        model_name="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 1. Search for context
    context = similarity_search(user_query)

    # 2. Build the system prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful assistant. Use the following context to answer: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 3. Format history for LangChain
    formatted_history = []
    for m in chat_history[-6:]:
        if m.role == "user":
            formatted_history.append(HumanMessage(content=m.content))
        else:
            formatted_history.append(AIMessage(content=m.content))

    # 4. Run the chain
    chain = prompt | llm
    response = chain.invoke({
        "input": user_query,
        "chat_history": formatted_history
    })
    
    return response.content