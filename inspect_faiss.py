"""
Script to inspect Faiss vector store data
"""
import pickle
import faiss
import numpy as np
from pathlib import Path

def inspect_vector_store(index_dir="local_pdf_index"):
    """Inspect the contents of a Faiss vector store"""
    index_path = Path(index_dir)
    
    # Load the pickle file containing metadata
    with open(index_path / "index.pkl", "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    
    # Load the Faiss index
    index = faiss.read_index(str(index_path / "index.faiss"))
    
    # Print basic information about the vector store
    print("\n=== Vector Store Information ===")
    print(f"Total vectors: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
    print(f"Index type: {type(index).__name__}")
    
    # Print information about the document store
    print("\n=== Document Store Information ===")
    print(f"Number of documents: {len(docstore._dict)}")
    
    # Print sample documents
    print("\n=== Sample Documents ===")
    for i, (doc_id, doc) in enumerate(list(docstore._dict.items())[:3]):
        print(f"\nDocument {i+1}:")
        print(f"ID: {doc_id}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:200]}...")
    
    # Print information about the index mapping
    print("\n=== Index to Document ID Mapping ===")
    print(f"Number of mappings: {len(index_to_docstore_id)}")
    print("\nSample mappings (first 5):")
    for i, (index_id, doc_id) in enumerate(list(index_to_docstore_id.items())[:5]):
        print(f"Vector index {index_id} -> Document ID: {doc_id}")
    
    # Calculate some statistics about the vectors
    if index.ntotal > 0:
        # Get all vectors
        vectors = faiss.vector_to_array(index.get_xb())
        vectors = vectors.reshape(index.ntotal, index.d)
        
        print("\n=== Vector Statistics ===")
        print(f"Vector shape: {vectors.shape}")
        print(f"Mean vector norm: {np.linalg.norm(vectors, axis=1).mean():.4f}")
        print(f"Mean vector: {vectors.mean(axis=0)[:5]}...")  # Show first 5 dimensions
        print(f"Vector standard deviation: {vectors.std(axis=0)[:5]}...")  # Show first 5 dimensions

if __name__ == "__main__":
    inspect_vector_store()
