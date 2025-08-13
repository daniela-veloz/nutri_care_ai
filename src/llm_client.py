import os
from langchain_openai import ChatOpenAI
from src.environment_loader import EnvironmentLoader


class LLMClient:
    """Singleton client for managing OpenAI ChatGPT language model connections.
    
    This class provides a centralized way to create and access a ChatOpenAI client
    for the nutrition care AI application. It implements the singleton pattern to
    ensure only one instance of the LLM client exists throughout the application
    lifecycle, which helps manage API rate limits and connection pooling.
    
    The client automatically handles API key retrieval from environment variables
    and supports configurable model selection for different use cases in the
    RAG (Retrieval-Augmented Generation) workflow.
    """
    _llm_client = None

    @staticmethod
    def get_llm_client() -> ChatOpenAI:
        """Get or create a singleton instance of the ChatOpenAI client.
        
        This method implements lazy initialization of the ChatOpenAI client.
        On first call, it retrieves the OpenAI API key from environment variables,
        configures the model settings, and creates the client instance. Subsequent
        calls return the existing instance.
        
        Returns:
            ChatOpenAI: A configured ChatOpenAI client instance ready for use.
            
        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set
                       and cannot be loaded from the environment configuration.
            Exception: If there are issues initializing the ChatOpenAI client
                      (e.g., invalid API key, network connectivity issues).
        
        Note:
            The default model is 'gpt-4o-mini' which provides a good balance of
            performance and cost-effectiveness for nutrition care applications.
            This can be overridden by setting the LLM_MODEL environment variable.
        """
        if LLMClient._llm_client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                api_key = EnvironmentLoader.get_required_env('OPENAI_API_KEY')
            
            llm_model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
            
            LLMClient._llm_client = ChatOpenAI(
                openai_api_key=api_key,
                model=llm_model,
                streaming=False
            )
        
        return LLMClient._llm_client