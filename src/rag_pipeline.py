from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_vector_store(chunks: List[Document], api_key: str) -> Optional[FAISS]:
    """
    Creates a FAISS vector store from the given text chunks using Google Generative AI embeddings.
    """
    if not chunks:
        return None
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", 
        google_api_key=api_key
    )
    
  
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vector_store

from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_chain(vector_store: FAISS, api_key: str) -> Optional[Any]:
    """
    Sets up the Retrieval-Augmented Generation (RAG) chain using LangChain Expression Language (LCEL).
    
    Args:
        vector_store (FAISS): The populated vector database.
        api_key (str): The Google API key for the LLM.
        
    Returns:
        Runnable: The configured LCEL chain ready to be invoked, or None if vector_store is missing.
    """
    if not vector_store:
        return None
        
    # Setup the retriever to get top 5 chunks (as per notebook)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Define the prompt template (combining assignment requirement with notebook style)
    prompt_template = ChatPromptTemplate.from_template(
        """
You are an AI assistant.

If the user is simply greeting you (e.g., "hi", "hello"), respond politely and ask how you can help them with their documents. 

For all other questions, answer only using the provided context.
If the answer is unavailable in the context, say exactly:
"I couldn't find this information in the uploaded documents."

Context:
{context}

Question:
{question}
        """
    )
    
    # Initialize the LLM (from notebook)
    llm = GoogleGenerativeAI(
        model="gemini-3.5-flash-lite", 
        google_api_key=api_key
    )
    
    # Construct the LCEL chain with explicit document formatting and string output parsing
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return chain

def generate_answer(chain: Any, query: str) -> str:
    """
    Invokes the LCEL chain to generate an answer for the user's query based on the vector store context.
    
    Args:
        chain (Runnable): The active RAG chain.
        query (str): The user's question.
        
    Returns:
        str: The AI's response.
    """
    if not chain:
        return "System is not initialized. Please upload documents and provide an API key."
        
    return chain.invoke(query)
