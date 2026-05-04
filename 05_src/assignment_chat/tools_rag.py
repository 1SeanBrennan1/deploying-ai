# tools_rag.py
from langchain.tools import tool
import chromadb
import pandas as pd
import os
from dotenv import load_dotenv
from utils.logger import get_logger

_logs = get_logger(__name__)

# Determine the directory of this file
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .secrets from the parent directory (05_src)
load_dotenv(os.path.join(_TOOLS_DIR, "..", ".secrets"))

# Build paths relative to this file's location
CHROMA_PATH = os.path.join(_TOOLS_DIR, "chroma_data")
CSV_DATA_PATH = os.path.join(_TOOLS_DIR, "data", "my_data.csv")

os.makedirs(CHROMA_PATH, exist_ok=True)

# Setup ChromaDB with file persistence using the centralized embedding config
from assignment_chat.llm_config import get_embedding_function

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or get the collection
collection = chroma_client.get_or_create_collection(
    name="my_knowledge_base",
    embedding_function=get_embedding_function()
)


@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches our internal knowledge base for information matching the query.
    
    Use this tool when users ask questions that might be answered by our 
    internal documents, articles, or reference materials.
    
    Args:
        query: The search query or question from the user
    
    Returns:
        Formatted search results with relevant information
    """
    _logs.info(f'Searching knowledge base for: {query}')
    
    # Step 1: Perform semantic search in ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    if not results['documents'][0]:
        return "I searched through my knowledge base but couldn't find anything relevant to your question. Could you try rephrasing it?"
    
    # Step 2: Get additional details from CSV for each result
    enriched_results = []
    for idx, doc_id in enumerate(results['ids'][0]):
        document_text = results['documents'][0][idx]
        metadata = results['metadatas'][0][idx]
        additional_info = get_additional_details_from_csv(doc_id)
        
        enriched_results.append({
            "text": document_text,
            "source": metadata.get("source", "Unknown Source"),
            "title": additional_info.get("title", "Untitled"),
            "author": additional_info.get("author", "Unknown Author"),
            "category": additional_info.get("category", "General")
        })
    
    # Step 3: Format results in natural language
    return format_search_results(enriched_results)


def get_additional_details_from_csv(doc_id: str) -> dict:
    """
    Reads additional metadata from a CSV file to enrich search results.
    
    Args:
        doc_id: The document ID to look up in the CSV
    
    Returns:
        Dictionary with additional details about the document
    """
    try:
        if not os.path.exists(CSV_DATA_PATH):
            _logs.warning(f'CSV data file not found at {CSV_DATA_PATH}')
            return {}
        
        df = pd.read_csv(CSV_DATA_PATH)
        
        # Extract the numeric ID from the document ID (e.g., "doc_5" -> 5)
        numeric_id = get_id_from_doc_id(doc_id)
        
        if numeric_id is None:
            return {}
        
        # Look up the row in the CSV
        row = df[df['id'] == numeric_id]
        
        if row.empty:
            return {}
        
        return {
            "title": row.iloc[0].get("title", "Untitled"),
            "author": row.iloc[0].get("author", "Unknown Author"),
            "category": row.iloc[0].get("category", "General"),
            "date": row.iloc[0].get("date", "Unknown Date")
        }
    except Exception as e:
        _logs.error(f'Error reading CSV data: {str(e)}')
        return {}


def get_id_from_doc_id(doc_id: str) -> int:
    """
    Extracts the numeric ID from a document ID string.
    For example, "doc_5" becomes 5.
    
    Args:
        doc_id: The document ID string like "doc_5"
    
    Returns:
        The numeric ID, or None if extraction fails
    """
    try:
        parts = doc_id.split('_')
        if len(parts) >= 2:
            return int(parts[1])
        return None
    except (ValueError, IndexError):
        return None


def format_search_results(results: list) -> str:
    """
    Formats search results into a natural language response.
    
    Args:
        results: List of enriched result dictionaries
    
    Returns:
        A formatted natural language response
    """
    if not results:
        return "I couldn't find anything matching your query in my knowledge base."
    
    response = "Here's what I found in my knowledge base:\n\n"
    
    for i, result in enumerate(results, 1):
        response += f"📚 Result {i}:\n"
        
        if result.get("title"):
            response += f"   Title: {result['title']}\n"
        
        if result.get("author"):
            response += f"   Author: {result['author']}\n"
        
        if result.get("source"):
            response += f"   Source: {result['source']}\n"
        
        if len(result['text']) > 500:
            response += f"   Content: {result['text'][:500]}...\n"
        else:
            response += f"   Content: {result['text']}\n"
        
        response += "\n"
    
    response += "If you'd like more details on any of these, just let me know!"
    
    return response