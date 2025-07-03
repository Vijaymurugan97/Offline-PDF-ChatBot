"""
Script to demonstrate querying and analyzing the Faiss vector store with focused document sections
"""
import pickle
import faiss
import numpy as np
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings

def find_document_section(docstore, section_keywords):
    """Find document sections containing specific keywords"""
    matching_docs = []
    for doc_id, doc in docstore._dict.items():
        content = doc.page_content.lower()
        if any(keyword.lower() in content for keyword in section_keywords):
            matching_docs.append((doc_id, doc))
    return matching_docs

def analyze_vector_store(index_dir="local_pdf_index"):
    """Analyze and query the vector store with focus on key document sections"""
    index_path = Path(index_dir)
    
    # Load the pickle file containing metadata
    print("\nLoading vector store...")
    with open(index_path / "index.pkl", "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    
    # Search for document purpose
    print("\n=== Document Purpose ===")
    purpose_docs = find_document_section(docstore, ["purpose", "summary", "introduction"])
    for _, doc in sorted(purpose_docs, key=lambda x: x[1].metadata.get('page', 999))[:2]:
        if "<SUMMARY" in doc.page_content:
            print(f"Page {doc.metadata.get('page', 'unknown')}:")
            content = doc.page_content.replace("<SUMMARY_BEGIN>", "").replace("<SUMMARY_END>", "").strip()
            print(content)
            print("-" * 80)
    
    # Search for interactivity rules
    print("\n=== Interactivity Rules ===")
    rules_docs = find_document_section(docstore, ["rule:", "❖ rule", "specific rules"])
    for _, doc in sorted(rules_docs, key=lambda x: x[1].metadata.get('page', 999))[:3]:
        if "rule" in doc.page_content.lower():
            print(f"Page {doc.metadata.get('page', 'unknown')}:")
            # Clean up and format the content
            content = doc.page_content.strip()
            content = ' '.join([line.strip() for line in content.split('\n') if line.strip()])
            if "❖ Rule:" in content:
                rules = content.split("❖ Rule:")
                for rule in rules[1:]:  # Skip the first split as it's before the first rule
                    print(f"Rule: {rule.split('Example:')[0].strip()}")
            print("-" * 80)
    
    # Search for schematic handling
    print("\n=== Schematic Handling ===")
    schematic_docs = find_document_section(docstore, ["schematic diagram", "wiring diagram", "navigation"])
    for _, doc in sorted(schematic_docs, key=lambda x: x[1].metadata.get('page', 999))[:3]:
        if any(term in doc.page_content.lower() for term in ["navigation", "schematic", "wiring"]):
            print(f"Page {doc.metadata.get('page', 'unknown')}:")
            content = doc.page_content.strip()
            content = ' '.join([line.strip() for line in content.split('\n') if line.strip()])
            # Remove header/footer boilerplate
            if "Airbus Amber" in content:
                content = content.split("Portal .")[1] if "Portal ." in content else content
            print(content)
            print("-" * 80)

if __name__ == "__main__":
    analyze_vector_store()
