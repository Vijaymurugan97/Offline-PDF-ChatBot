"""
Offline PDF QA Chatbot using LangChain and local LLMs
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.chains import RetrievalQA
from langchain.schema import Document

def validate_files(pdf_path: str, model_path: str) -> None:
    """Validate that required files exist."""
    if not Path(pdf_path).is_file():
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    if not Path(model_path).is_file():
        print(f"Error: LLM model file not found at {model_path}")
        sys.exit(1)

def extract_section_title(text: str) -> str:
    """Extract section title from text based on common patterns."""
    markers = ["TAG NAME", "HOTSPOT CODE", "Rule:", "❖"]
    for marker in markers:
        if marker in text:
            lines = text.split('\n')
            for line in lines:
                if marker in line:
                    return line.strip()
    return ""

def enhance_chunk_metadata(chunk: Document, section_title: str, page_num: int) -> Dict:
    """Enhance chunk metadata with section information and page numbers."""
    metadata = chunk.metadata.copy()
    metadata.update({
        "page": page_num,
        "section": section_title,
        "source_type": "technical_document",
        "document_type": "specification",
        "has_table": "│" in chunk.page_content or "+" in chunk.page_content,
        "has_rule": "Rule:" in chunk.page_content or "❖" in chunk.page_content
    })
    return metadata

def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    """Load and chunk the PDF file with enhanced section awareness."""
    print(f"Loading PDF from: {pdf_path}")
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". "],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        enhanced_chunks = []
        for doc in documents:
            section_title = extract_section_title(doc.page_content)
            chunks = text_splitter.split_text(doc.page_content)
            
            for chunk in chunks:
                enhanced_chunk = Document(
                    page_content=chunk,
                    metadata=enhance_chunk_metadata(
                        doc,
                        section_title,
                        doc.metadata.get("page", 0)
                    )
                )
                enhanced_chunks.append(enhanced_chunk)

        print(f"Split into {len(enhanced_chunks)} enhanced chunks.")
        return enhanced_chunks
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        sys.exit(1)

def create_vector_store(doc_chunks: List[Document], embeddings, persist_dir="local_pdf_index"):
    """Create or load a vector store with enhanced search capabilities."""
    try:
        # Clear existing vector store if it exists
        if os.path.exists(persist_dir):
            print(f"Clearing existing vector store at {persist_dir}...")
            shutil.rmtree(persist_dir)
        
        print("Creating new vector store...")
        vector_store = FAISS.from_documents(doc_chunks, embeddings)
        vector_store.save_local(persist_dir)
        print(f"Vector store saved to {persist_dir}.")
        return vector_store
    except Exception as e:
        print(f"Error with vector store: {str(e)}")
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        vector_store = FAISS.from_documents(doc_chunks, embeddings)
        vector_store.save_local(persist_dir)
        return vector_store

def load_llm(model_path: str, model_type="llama"):
    """Load the local LLM model."""
    print(f"Loading local LLM model from: {model_path}")
    try:
        abs_model_path = os.path.abspath(model_path)
        print(f"Using absolute model path: {abs_model_path}")
        
        if model_type == "llama":
            llm = LlamaCpp(
                model_path=abs_model_path,
                temperature=0.75,
                max_tokens=2000,
                n_ctx=2048,
                top_p=1,
                verbose=True
            )
        else:  # gpt4all
            llm = GPT4All(
                model=abs_model_path,
                n_threads=8,
                verbose=True
            )
        return llm
    except Exception as e:
        print(f"Error loading LLM model: {str(e)}")
        sys.exit(1)

def enhance_query(query: str) -> str:
    """Enhance the query with explicit keyword guidance."""
    if "tag" in query.lower() or "hotspot" in query.lower():
        return f"Find exact TAG NAME or HOTSPOT CODE for: {query}"
    if "rule" in query.lower():
        return f"Find specific Rule or ❖ statement for: {query}"
    return query

def main():
    # User configuration
    pdf_path = "TDG0025.2_Issue K.pdf"
    model_path = "llama-2-7b-chat.Q4_K_M.gguf"  # Default to Llama model
    
    validate_files(pdf_path, model_path)

    # Enhanced document processing
    doc_chunks = load_and_chunk_pdf(pdf_path)

    # Initialize embeddings
    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create enhanced vector store (will clear existing store)
    vector_store = create_vector_store(doc_chunks, embeddings)

    # Load LLM
    llm = load_llm(model_path, model_type="llama")

    # Create enhanced retriever and QA chain
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("\nEnhanced PDF QA Chatbot is ready. Ask questions about the document (type 'exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() in {"exit", "quit"}:
            print("Exiting chatbot. Goodbye!")
            break
            
        # Enhance the query
        enhanced_query = enhance_query(query)
        
        # Get response
        result = qa_chain({"query": enhanced_query})
        answer = result["result"]
        source_docs = result["source_documents"]
        
        print("\nAnswer:\n", answer)
        print("\nSource document chunks used:")
        for i, doc in enumerate(source_docs):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Section: {doc.metadata.get('section', 'N/A')}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print("Content:", doc.page_content)
        print("-" * 40)

if __name__ == "__main__":
    main()
