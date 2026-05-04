# llm_config.py
"""
Centralized LLM configuration for the Sage agent.
Update this ONE file to change models, endpoints, or API keys everywhere.
"""

import os
from dotenv import load_dotenv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, "..", ".secrets"))

# ---- OpenAI / LangChain Chat Model Config ----
OPENAI_CHAT_CONFIG = {
    "model": "openai:gpt-4o-mini",
    "base_url": "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    "api_key": "any value",
    "default_headers": {"x-api-key": os.getenv("API_GATEWAY_KEY")},
}

# ---- Embedding Config (ChromaDB) ----
EMBEDDING_CONFIG = {
    "api_key": "any value",
    "model_name": "text-embedding-3-small",
    "api_base": "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    "default_headers": {"x-api-key": os.getenv("API_GATEWAY_KEY")},
}


def get_chat_model(temperature: float = 0.0):
    """Returns a LangChain chat model using the centralized config."""
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        model=OPENAI_CHAT_CONFIG["model"],
        base_url=OPENAI_CHAT_CONFIG["base_url"],
        api_key=OPENAI_CHAT_CONFIG["api_key"],
        default_headers=OPENAI_CHAT_CONFIG["default_headers"],
        temperature=temperature,
    )


def get_embedding_function():
    """Returns an OpenAI embedding function for ChromaDB using centralized config."""
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    return OpenAIEmbeddingFunction(
        api_key=EMBEDDING_CONFIG["api_key"],
        model_name=EMBEDDING_CONFIG["model_name"],
        api_base=EMBEDDING_CONFIG["api_base"],
        default_headers=EMBEDDING_CONFIG["default_headers"],
    )