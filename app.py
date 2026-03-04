import streamlit as st
import os
from src.document_processor import process_pdf
from src.vector_store import add_documents_to_store, get_vector_store
from src.rag_pipeline import build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Nexus", layout="wide")

# --- Custom Dark Sleek CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 50%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Top bar fix */
    header { background: transparent !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Sleek Sidebar */
    .css-1544g2n, [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 15, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    hr {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 2rem 0;
    }
    
    /* Eye-Catching Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 198, 255, 0.5);
    }
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    /* Dark Glass Inputs */
    .stTextInput>div>div>input, .stSelectbox>div>div>select, [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        box-shadow: none !important;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 0 1px #00f2fe !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Floating Chat Bubbles */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
    }
    
    /* User message neon accent */
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: rgba(0, 242, 254, 0.05);
        border-left: 3px solid #00f2fe;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="chat-message-assistant"] {
        border-left: 3px solid #8e2de2;
    }
    
    /* Docked Chat Input */
    .stChatInputContainer {
        border-radius: 50px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        background-color: rgba(10, 10, 10, 0.9) !important;
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.5) !important;
        padding: 4px 10px !important;
    }
    
    /* Styled Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px;
        color: #a0aec0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        background-color: transparent;
    }
    
    /* Intro text */
    .intro-text {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 300;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Nexus AI")
st.markdown("<p class='intro-text'>Seamless synthesis and retrieval.</p>", unsafe_allow_html=True)

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
