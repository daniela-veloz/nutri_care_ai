"""
Mem0 Client

Centralized client for managing Mem0 memory service connections for customer
interaction storage and retrieval in the nutrition care AI system.
"""

import os
from mem0 import MemoryClient
from src.environment_loader import EnvironmentLoader


class Mem0Client:
    """Singleton client for managing Mem0 memory service connections.
    
    This class provides a centralized way to create and access a Mem0 MemoryClient
    for the nutrition care AI application. It implements the singleton pattern to
    ensure only one instance of the memory client exists throughout the application
    lifecycle, which helps manage API connections and memory operations efficiently.
    
    The client automatically handles API key retrieval from environment variables
    and provides a consistent interface for storing and retrieving customer
    interactions, conversation history, and contextual information that enhances
    the personalization of nutrition care responses.
    
    Mem0 enables the system to maintain conversation continuity across multiple
    interactions by storing relevant context and retrieving it when needed for
    more personalized and context-aware responses.
    """
    _mem0_client = None

    @staticmethod
    def get_mem0_client() -> MemoryClient:
        """Get or create a singleton instance of the Mem0 MemoryClient.
        
        This method implements lazy initialization of the Mem0 MemoryClient.
        On first call, it retrieves the Mem0 API key from environment variables,
        and creates the client instance. Subsequent calls return the existing instance.
        
        The method uses a fallback approach where it first checks for the API key
        in the current environment, and if not found, uses the centralized
        EnvironmentLoader to retrieve the required key.
        
        Returns:
            MemoryClient: A configured Mem0 MemoryClient instance ready for use.
            
        Raises:
            ValueError: If the MEM0_API_KEY environment variable is not set
                       and cannot be loaded from the environment configuration.
            Exception: If there are issues initializing the MemoryClient
                      (e.g., invalid API key, network connectivity issues).
        
        Note:
            The MemoryClient is used for storing customer interactions,
            conversation context, and retrieving relevant historical information
            to provide personalized nutrition care responses.
        """
        if Mem0Client._mem0_client is None:
            api_key = os.getenv('MEM0_API_KEY')
            if not api_key:
                api_key = EnvironmentLoader.get_required_env('MEM0_API_KEY')
            
            Mem0Client._mem0_client = MemoryClient(api_key)
        
        return Mem0Client._mem0_client