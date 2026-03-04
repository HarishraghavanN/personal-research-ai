import streamlit as st
import os
from src.document_processor import process_pdf
from src.vector_store import add_documents_to_store, get_vector_store
from src.rag_pipeline import build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Personal Research AI", layout="wide")
st.title("📚 Personal Research AI")
st.write("Upload PDFs and ask questions based on their content.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    llm_provider = st.selectbox("LLM Provider", ["Google Gemini", "OpenAI", "Anthropic Claude"])
    
    # Store API key in session state for UI check, but also set env var for Langchain
    api_key = ""
    if llm_provider == "Google Gemini":
        api_key = st.text_input("Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    elif llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    elif llm_provider == "Anthropic Claude":
        api_key = st.text_input("Anthropic API Key", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader("Upload research PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing documents (Chunking & Embedding)..."):
                try:
                    all_chunks = []
                    for file in uploaded_files:
                        chunks = process_pdf(file)
                        all_chunks.extend(chunks)
                    
                    # Ingest into vector store
                    st.session_state.vector_store = add_documents_to_store(all_chunks)
                    st.success(f"Successfully processed {len(all_chunks)} chunks into the vector store!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

# --- Initialize Session State ---
if "vector_store" not in st.session_state:
    try:
        # Load local chroma db if exists and we can
        # For a more robust app, we'd check if the directory is valid
        if os.path.exists("./chroma_db"):
            st.session_state.vector_store = get_vector_store()
        else:
            st.session_state.vector_store = None
    except Exception as e:
        st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Interface ---
# Display chat history
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Add user message to UI immediately
    with st.chat_message("user"):
        st.markdown(user_query)
        
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
    elif st.session_state.vector_store is None:
        st.error("Vector store is empty. Please upload and process documents first.")
    else:
        # Build chain and generate response
        rag_chain = build_rag_chain(st.session_state.vector_store)
        if rag_chain is None:
            st.error("Failed to initialize the LLM. Check your API key.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching documents & generating answer..."):
                    try:
                        response = rag_chain.invoke({
                            "input": user_query,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = response["answer"]
                        
                        st.markdown(answer)
                        
                        # Add citations (sources)
                        with st.expander("View Source Documents"):
                            sources = response.get("context", [])
                            for idx, doc in enumerate(sources):
                                source_name = doc.metadata.get("source", "Unknown")
                                page_num = doc.metadata.get("page", "Unknown")
                                st.write(f"**Source {idx+1}:** {source_name} (Page {page_num})")
                                st.caption(doc.page_content[:200] + "...")
                        
                        # Update short-term memory
                        st.session_state.chat_history.extend(
                            [HumanMessage(content=user_query), AIMessage(content=answer)]
                        )
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
