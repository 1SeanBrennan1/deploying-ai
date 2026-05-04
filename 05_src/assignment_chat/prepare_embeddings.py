# prepare_embeddings.py
"""
Run this script ONCE to prepare your knowledge base for semantic search.

This script:
1. Reads the CSV data file
2. Creates embeddings for each document using OpenAI's text-embedding-3-small
3. Stores the embeddings in ChromaDB with file persistence

After running this, your RAG tool (search_knowledge_base) will work.
"""

import chromadb
import pandas as pd
import os
from dotenv import load_dotenv

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .secrets from the parent directory (05_src)
load_dotenv(os.path.join(SCRIPT_DIR, "..", ".secrets"))

# Build absolute paths from the script location
CSV_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "my_data.csv")
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma_data")

# Use the centralized embedding config - ONLY place to change providers/endpoints
from assignment_chat.llm_config import get_embedding_function


def prepare_embeddings():
    """Main function to prepare and store embeddings."""
    
    print("📚 Starting knowledge base preparation...")
    
    # Check if CSV exists
    if not os.path.exists(CSV_DATA_PATH):
        print(f"Error: CSV file not found at {CSV_DATA_PATH}")
        print("Please create your data/my_data.csv file first.")
        return
    
    # Load the CSV data
    print("📖 Loading data from CSV...")
    df = pd.read_csv(CSV_DATA_PATH)
    print(f"   Found {len(df)} documents.")
    
    # Create ChromaDB directory if it doesn't exist
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Initialize ChromaDB with persistence
    print("🔧 Setting up ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Delete existing collection if it exists (to start fresh)
    try:
        chroma_client.delete_collection("my_knowledge_base")
        print("   Deleted existing collection.")
    except Exception:
        pass
    
    # Create a new collection using the centralized embedding function
    collection = chroma_client.create_collection(
        name="my_knowledge_base",
        embedding_function=get_embedding_function()
    )
    
    # Add documents to ChromaDB
    print("Creating embeddings and storing in ChromaDB...")
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        documents.append(row['text_column'])
        metadatas.append({
            "source": row.get('title', 'Untitled'),
            "author": row.get('author', 'Unknown'),
            "category": row.get('category', 'General'),
            "date": str(row.get('date', 'Unknown'))
        })
        ids.append(f"doc_{row['id']}")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully stored {len(documents)} documents in ChromaDB!")
    print(f"Storage location: {CHROMA_PATH}")
    print("\nYour knowledge base is ready! You can now use the search_knowledge_base tool.")


if __name__ == "__main__":
    prepare_embeddings()