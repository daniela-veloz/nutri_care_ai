import os
from langchain_openai import OpenAIEmbeddings
from src.environment_loader import EnvironmentLoader


class EmbeddingModel:
    """Singleton manager for OpenAI text embedding models in the RAG workflow.
    
    This class provides centralized access to OpenAI's text embedding models,
    which are essential for converting text documents and queries into vector
    representations. These embeddings enable semantic similarity search and
    retrieval in the nutrition care AI application.
    
    The class implements the singleton pattern to ensure consistent embedding
    model usage across all components of the RAG pipeline, including document
    ingestion, query processing, and context retrieval.
    
    The default model 'text-embedding-ada-002' provides high-quality embeddings
    with good performance characteristics for nutrition and healthcare domains.
    """
    _embedding_model = None

    @staticmethod
    def get_embedding_model() -> OpenAIEmbeddings:
        """Get or create a singleton instance of the OpenAI embeddings model.
        
        This method implements lazy initialization of the OpenAI embeddings model.
        On first call, it retrieves the API key, configures the model settings,
        and creates the embeddings instance. Subsequent calls return the existing
        instance, ensuring consistency across the application.
        
        Returns:
            OpenAIEmbeddings: A configured OpenAI embeddings instance ready for
                             generating vector representations of text.
        
        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set
                       and cannot be loaded from the environment configuration.
            Exception: If there are issues initializing the OpenAI embeddings
                      client (e.g., invalid API key, network connectivity issues).
        
        Note:
            The default embedding model is 'text-embedding-ada-002' which provides
            excellent semantic understanding for nutrition and healthcare content.
            This can be overridden by setting the EMBEDDING_MODEL environment variable.
        """
        if EmbeddingModel._embedding_model is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                api_key = EnvironmentLoader.get_required_env('OPENAI_API_KEY')
            
            embedding_model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
            
            EmbeddingModel._embedding_model = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=embedding_model_name
            )
        
        return EmbeddingModel._embedding_model