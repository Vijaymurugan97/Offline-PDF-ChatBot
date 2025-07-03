"""
Streamlit-based UI for the PDF QA Chatbot with enhanced model selection
"""

import streamlit as st
import os
import shutil
from pathlib import Path
import tempfile
from app import load_and_chunk_pdf, enhance_query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp, GPT4All
from langchain.chains import RetrievalQA
from models import get_available_local_models, ModelInfo, get_model_path


st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        background-color: #f7f7f8;
    }
    .user-message {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #f7f7f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #e7f3fe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

def load_llm(model_info: ModelInfo):
    """Load the selected LLM model."""
    try:
        # Get the correct model path
        model_path = get_model_path(model_info)
        if model_path is None:
            raise FileNotFoundError(f"Model file '{model_info.filename}' not found in models directory or current directory")
        
        st.info(f"Loading model from: {model_path}")
        
        if model_info.type == "llama":
            return LlamaCpp(
                model_path=str(model_path),
                temperature=0.75,
                max_tokens=2000,
                n_ctx=2048,
                top_p=1,
                verbose=True
            )
        else:  # gpt4all
            return GPT4All(
                model=str(model_path),
                n_threads=8,
                verbose=True,
                allow_download=False,  # Prevent automatic downloads
                model_path=str(model_path),  # Explicitly set model path
                model_type="gptj"  # Specify model type for GPT4All
            )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Model path attempted: {model_path}")
        return None

def clear_vector_store(persist_dir="local_pdf_index"):
    """Clear the existing vector store."""
    try:
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            print(f"Cleared vector store at {persist_dir}")
    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")

def create_vector_store(doc_chunks, embeddings, persist_dir="local_pdf_index"):
    """Create a new vector store."""
    try:
        print("Creating new vector store...")
        vector_store = FAISS.from_documents(doc_chunks, embeddings)
        vector_store.save_local(persist_dir)
        print(f"Vector store saved to {persist_dir}.")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def initialize_chatbot(pdf_path, model_info: ModelInfo):
    """Initialize the chatbot with the uploaded PDF and selected model."""
    try:
        with st.spinner('Processing PDF...'):
            # Clear existing vector store
            clear_vector_store()
            
            # Load and process the PDF
            doc_chunks = load_and_chunk_pdf(pdf_path)
            
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create vector store
            st.session_state.vector_store = create_vector_store(doc_chunks, embeddings)
            if not st.session_state.vector_store:
                st.error("Failed to create vector store")
                return
            
            # Load LLM model
            with st.status(f"Loading {model_info.name}...") as status:
                try:
                    model_path = get_model_path(model_info)
                    if model_path is None:
                        st.error(f"Model file not found: {model_info.filename}")
                        return
                    
                    llm = load_llm(model_info)
                    if not llm:
                        return
                    
                    # Create QA chain
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True
                    )
                    status.update(label=f"{model_info.name} loaded successfully!", state="complete")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")

def display_chat_history():
    """Display the chat history with alternating backgrounds."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>{message["content"]}
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                <div class="bot-message">
                    <strong>Assistant:</strong><br>{message["content"]}
                </div>
                """, unsafe_allow_html=True)
                if "sources" in message:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            **Source {i+1}** (Page {source.metadata.get('page', 'N/A')})  
                            Section: {source.metadata.get('section', 'N/A')}  
                            {source.page_content}
                            """)

def handle_user_input(user_question):
    """Handle user input and generate response."""
    try:
        with st.spinner('Thinking...'):
            # Enhance query
            enhanced_query = enhance_query(user_question)
            
            # Get response
            result = st.session_state.qa_chain({"query": enhanced_query})
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": source_docs
            })
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def display_model_info(model: ModelInfo):
    """Display detailed information about a model."""
    st.markdown(f"""
    <div class="model-info">
        <h4>{model.name}</h4>
        <p><strong>Description:</strong> {model.description}</p>
        <p><strong>Size:</strong> {model.size}</p>
        <p><strong>Best for:</strong> {', '.join(model.recommended_for)}</p>
        <p><strong>Type:</strong> {model.type}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("📚 PDF ChatBot")
    init_session_state()
    
    # Get available models
    available_models = get_available_local_models()
    if not available_models:
        st.error("No models found!")
        st.info("""
        Please ensure your models are in one of these locations:
        1. ./models/ directory (recommended)
        2. Current working directory
        """)
        return
    
    # Sidebar for PDF upload and model selection
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        st.subheader("Select Model")
        model_names = [model.name for model in available_models]
        selected_model_name = st.selectbox(
            "Choose a model:",
            model_names,
            index=0 if model_names else None,
            help="Select the AI model to use for processing your questions"
        )
        
        # Find selected model info
        selected_model = next(
            (model for model in available_models if model.name == selected_model_name),
            None
        )
        
        if selected_model:
            display_model_info(selected_model)
            model_path = get_model_path(selected_model)
            if model_path:
                st.success(f"Model file found at: {model_path}")
            else:
                st.error(f"Model file not found: {selected_model.filename}")
        
        st.markdown("---")
        
        # PDF upload
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file and selected_model:
            # Check if PDF or model selection changed
            if (st.session_state.current_pdf_name != uploaded_file.name or 
                st.session_state.selected_model != selected_model_name):
                # Clear chat history and vector store for new PDF or model
                st.session_state.chat_history = []
                st.session_state.current_pdf_name = uploaded_file.name
                st.session_state.selected_model = selected_model_name
                
                # Save uploaded file
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        st.session_state.uploaded_file_path = tmp_file.name
                    
                    # Initialize chatbot with selected model
                    initialize_chatbot(st.session_state.uploaded_file_path, selected_model)
                    st.success("PDF processed successfully! You can now start chatting.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        
        # Display information
        st.markdown("---")
        st.markdown("""
        ### How to use
        1. Select an AI model based on your needs:
           - GPT4All-J Groovy: Fast, good for general Q&A
           - Llama 2 7B Chat: Well-balanced performance
           - GPT4All Falcon: Best for technical content
        2. Upload a PDF file
        3. Wait for processing
        4. Ask questions about the document
        5. View sources in expandable sections
        """)
    
    # Main chat interface
    if st.session_state.qa_chain:
        chat_col, info_col = st.columns([2, 1])
        
        with chat_col:
            st.markdown("### Chat")
            display_chat_history()
            
            # Chat input
            with st.container():
                with st.form(key="chat_form", clear_on_submit=True):
                    user_question = st.text_area("Type your question here:", key="user_input", height=100)
                    cols = st.columns([1, 6, 1])
                    with cols[1]:
                        submit_button = st.form_submit_button("Send", use_container_width=True)
                    
                    if submit_button and user_question:
                        handle_user_input(user_question)
                        st.rerun()
        
        with info_col:
            st.markdown("### Document Info")
            if st.session_state.uploaded_file_path:
                st.info(f"Current PDF: {st.session_state.current_pdf_name}")
                st.success(f"Using model: {st.session_state.selected_model}")
                
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
                
                st.markdown("#### Statistics")
                if st.session_state.vector_store:
                    st.markdown(f"- Total chunks: {len(st.session_state.vector_store.docstore._dict)}")
                st.markdown(f"- Chat messages: {len(st.session_state.chat_history)}")
    else:
        st.info("👈 Please select a model and upload a PDF file to start chatting!")

if __name__ == "__main__":
    main()
