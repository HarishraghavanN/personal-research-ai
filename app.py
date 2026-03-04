import streamlit as st
import os
from src.document_processor import process_pdf
from src.vector_store import add_documents_to_store, get_vector_store
from src.rag_pipeline import build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Nexus", layout="wide")

# --- Custom Minimalist CSS ---
st.markdown("""
<style>
    /* Minimalist typography & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp {
        background-color: #fafafa;
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Headers */
    h1, h2, h3 {
        color: #111111;
        font-weight: 500 !important;
        letter-spacing: -0.01em;
    }
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    
    /* Hide default Streamlit elements */
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Sleek Sidebar */
    .css-1544g2n, [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eaeaea;
    }
    
    hr {
        border-top: 1px solid #eaeaea;
        margin: 2rem 0;
    }
    
    /* Monochrome Buttons */
    .stButton>button {
        background-color: #ffffff;
        color: #111111;
        border: 1px solid #d4d4d4;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 400;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    .stButton>button:hover {
        border-color: #111111;
        color: #111111;
        background-color: #f5f5f5;
    }
    .stButton>button:active {
        background-color: #e5e5e5;
    }
    
    /* Clean Inputs */
    .stTextInput>div>div>input, .stSelectbox>div>div>select, [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        color: #111111 !important;
        box-shadow: none !important;
        font-size: 0.95rem;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
        border-color: #111111 !important;
    }
    
    /* Refined Chat Bubbles */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    /* User message subtle differentiation */
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #fcfcfc;
        border-left: 2px solid #a3a3a3;
    }
    
    /* Assistant message differentiation */
    .stChatMessage[data-testid="chat-message-assistant"] {
        border-left: 2px solid #111111;
    }
    
    /* Chat input area */
    .stChatInputContainer {
        border-radius: 6px !important;
        border: 1px solid #d4d4d4 !important;
        background-color: #ffffff !important;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.02) !important;
    }
    
    /* Clean Expanders */
    .streamlit-expanderHeader {
        background-color: #f9f9f9 !important;
        border-radius: 4px;
        font-weight: 500;
        color: #333;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #ebebeb;
        border-radius: 4px;
        background-color: #ffffff;
    }
    
    /* Subtle text */
    .stMarkdown p {
        color: #404040;
    }
</style>
""", unsafe_allow_html=True)

st.title("Nexus")
st.markdown("<p style='color: #666666; font-size: 1.05rem; font-weight: 300; margin-bottom: 2rem;'>Document analysis and synthesis.</p>", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    try:
        import langchain_google_genai
        st.caption(f"SDK Version: {langchain_google_genai.__version__}")
    except:
        pass
        
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
            
    # --- Diagnostics ---
    if st.button("Run API Diagnostics"):
        if not api_key:
            st.error("Please enter your API Key first.")
        elif llm_provider == "Google Gemini":
            import requests
            try:
                # Ask Google directly which models this specific API key is allowed to use
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                res = requests.get(url).json()
                if "models" in res:
                    valid_models = [m['name'].replace('models/', '') for m in res['models'] if 'generateContent' in m.get('supportedGenerationMethods', [])]
                    st.success(f"**Your API key has access to these generating models:**\n" + "\n".join([f"- `{m}`" for m in valid_models]))
                else:
                    st.error(f"API Error: {res}")
            except Exception as e:
                st.error(f"Failed to reach Google API: {e}")
            
    st.header("Document Upload")
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
