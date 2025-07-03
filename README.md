# Offline-PDF-ChatBot
An enhanced LangChain-based local AI system for accurate PDF document querying, featuring a modern web interface similar to ChatGPT. The system is specifically optimized for technical documentation and provides an intuitive user experience.

## Key Features

- **Modern Web Interface**: Web UI with Streamlit
- **PDF Upload**: Easy document upload functionality
- **Chat History**: Persistent conversation history with source tracking
- **Table/Section-Aware Chunking**: Intelligently splits documents based on natural section boundaries and table structures
- **Enhanced Metadata**: Tracks section titles, page numbers, and content types for better context
- **Semantic Search**: Improved retrieval using metadata and content-aware searching
- **Query Enhancement**: Automatically adds context for technical queries about tags, hotspots, and rules
- **Local Processing**: All processing happens offline using local models

## Requirements

- Python 3.8+
- Required packages:
  ```bash
  pip install langchain PyPDF2 sentence-transformers faiss-cpu gpt4all llama-cpp-python unstructured pymupdf streamlit
  ```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

4. Upload your PDF and start chatting!

## Enhanced Document Processing

1. **Intelligent Chunking**
   - Recognizes section boundaries
   - Preserves table structures
   - Maintains context between related content

2. **Metadata Enhancement**
   - Section titles
   - Page numbers
   - Content type detection (tables, rules, etc.)
   - Technical document context

3. **Improved Search**
   - Semantic search using enhanced metadata
   - Better handling of technical terms
   - Context-aware query processing

## Usage

### Web Interface

1. **Upload PDF**:
   - Click the upload button in the sidebar
   - Select your PDF file
   - Wait for processing completion

2. **Chat Interface**:
   - Type your questions in the input field
   - View responses with source citations
   - Expand source sections for detailed context

3. **Features**:
   - Clean, intuitive interface
   - Real-time response generation
   - Source tracking and display
   - Chat history persistence
   - Mobile-friendly design

### Command Line Interface

For those who prefer command line usage:

1. Run the basic version:
   ```bash
   python app.py
   ```
2. Follow the prompts to interact with the PDF

## Query Examples

- "What is the hotspot code for a wire symbol?"
- "What are the rules for VU and VE panels?"
- "Show me the tag structure for manufacturer data"

## Implementation Details

### Document Processing Pipeline

1. **Loading**: Uses PyMuPDFLoader for better PDF handling
2. **Chunking**: Section-aware splitting with RecursiveCharacterTextSplitter
3. **Metadata**: Enhanced with section information and content type
4. **Vectorization**: FAISS vector store with metadata filtering
5. **Retrieval**: Context-aware search with metadata filtering

### Query Processing

1. **Query Enhancement**: Adds technical context to queries
2. **Retrieval**: Uses metadata to improve search accuracy
3. **Response**: Includes source context and metadata

## Evaluation

The system includes built-in logging of:
- Query processing steps
- Retrieved chunks and their metadata
- Source document references
- Response generation

This helps in:
- Validating accuracy
- Identifying improvement areas
- Tracking system performance

## Future Improvements

1. Add support for multiple document types
2. Implement custom chunking strategies for different document structures
3. Add evaluation benchmarks using known Q&A pairs
4. Enhance metadata extraction for better context understanding



LangChain	Orchestration	Helps connect different components like document loaders, chunkers, vector DBs, and LLMs. Think of it like a manager for handling PDF→Query→Answer flow.
Streamlit	UI	A Python tool to build modern web apps with minimal code. This gives the project its Web interface.
PyMuPDF / PyPDF2	PDF Reading	These are libraries that read PDF files and extract text, images, page numbers, etc.
RecursiveCharacterTextSplitter	Chunking	Splits large PDF content into smaller, meaningful parts (e.g., sections, tables) so that the model can understand and answer questions accurately.
FAISS	Vector Database	Stores text chunks in a way that enables fast semantic search (similar meaning matching) instead of simple keyword search.
HuggingFace Transformers	Embeddings	Turns text into vectors (numbers) that the model can compare to find relevant content.
LLaMA / GPT4All / llama-cpp-python	LLMs	Local models that understand the question and generate responses, just like ChatGPT but fully offline.
unstructured	Text Parsing	Helps identify different content types like titles, tables, paragraphs from a PDF, useful for technical documents.
Sentence-Transformers	Embedding Generator	Used to create semantic vector representations of the text for better matching with questions.
