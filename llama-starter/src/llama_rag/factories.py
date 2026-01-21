"""
common/factories.py

Simple factory functions for creating LangChain components.
No config reading - parameters are passed explicitly from caller.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def get_embeddings(type: str = "ollama", model: str = "embeddinggemma") -> Embeddings:
    """
    Get embeddings instance.
    
    Args:
        type: Embedding type (ollama, openai, huggingface)
        model: Model name
    
    Returns:
        Embeddings instance
    """
    if type == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    
    elif type == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    
    elif type == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)
    
    else:
        raise ValueError(f"Unknown embedding type: {type}")


def get_llm(
    type: str = "ollama",
    model: str = "llama3.2",
    temperature: float = 0.7
) -> BaseChatModel:
    """
    Get LLM instance.
    
    Args:
        type: LLM type (ollama, openai, llamacpp)
        model: Model name
        temperature: Temperature for generation
    
    Returns:
        BaseChatModel instance
    """
    if type == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    
    elif type == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    
    elif type == "llamacpp":
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(model_path=model, temperature=temperature)
    
    else:
        raise ValueError(f"Unknown LLM type: {type}")


def get_vectorstore(embeddings: Embeddings, type: str = "faiss"):
    """
    Get vector store class.
    
    Args:
        embeddings: Embeddings instance
        type: Vector store type (faiss, chroma)
    
    Returns:
        VectorStore class (not instance)
    """
    if type == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS
    
    elif type == "chroma":
        from langchain_community.vectorstores import Chroma
        return Chroma
    
    else:
        raise ValueError(f"Unknown vectorstore type: {type}")
