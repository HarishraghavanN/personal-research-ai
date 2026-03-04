import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

def get_llm():
    # Attempt to load LLM based on environment variables
    # Clear accidental cross-contamination from Streamlit UI
    
    if os.getenv("GOOGLE_API_KEY"):
        # Gemini 1.5 is deprecated. Using the modern 2.5 / 2.0 models.
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True)
        except Exception:
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)

    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo")
    elif os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-haiku-20240307")
    else:
        # We will surface this in Streamlit if no key is found
        return None

def build_rag_chain(vector_store):
    llm = get_llm()
    if not llm:
        return None
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Get top 5 semantic chunks
    
    # Semantic Search & Short-Term Memory: Enhance the retriever with chat history context
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt
    system_prompt = (
        "You are a helpful Personal Research AI assistant. "
        "Use the following pieces of retrieved context to answer "
        "the user's question. If you don't know the answer, say that you "
        "don't know. Keep the answer detailed but concise. "
        "You MUST cite the source document metadata in your response "
        "(e.g., [Source: file_name.pdf, Page: 2]).\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
