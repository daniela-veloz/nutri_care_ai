"""
Environment Loader

Centralized environment variable loading for the Nutrition Care AI application.
This module ensures that environment variables are loaded once and can be safely
imported by multiple classes without conflicts.
"""

import os
from dotenv import load_dotenv


class EnvironmentLoader:
    """Singleton class for loading and managing environment variables.
    
    This class provides a centralized way to load environment variables from
    .env files and manage their access throughout the nutrition care AI application.
    It implements a singleton pattern to ensure environment variables are loaded
    only once, preventing conflicts and improving performance.
    
    The class supports both required and optional environment variables,
    providing appropriate error handling for missing required configurations.
    This is particularly important for API keys, database connections, and
    other critical configuration parameters needed for the RAG workflow.
    """
    _loaded = False
    
    @classmethod
    def load_environment(cls):
        """Load environment variables from .env file if not already loaded.
        
        This method uses the singleton pattern to ensure environment variables
        are loaded only once during the application lifecycle. It searches for
        .env files in the current directory and parent directories following
        the standard dotenv behavior.
        
        Note:
            This method is called automatically by other methods in this class,
            so explicit calls are typically not necessary.
        """
        if not cls._loaded:
            load_dotenv()
            cls._loaded = True
    
    @staticmethod
    def get_required_env(key: str) -> str:
        """eGet a required environment variable with validation.
        
        This method retrieves an environment variable that is essential for
        application functionality. If the variable is not set, it raises a
        ValueError with a descriptive message to help with debugging.
        
        Args:
            key: The name of the environment variable to retrieve.
            
        Returns:
            str: The value of the environment variable.
            
        Raises:
            ValueError: If the environment variable is not set or is empty.
                       The error message includes the variable name to aid
                       in troubleshooting configuration issues.
        
        """
        EnvironmentLoader.load_environment()
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is required")
        return value
    
    @staticmethod
    def get_optional_env(key: str, default: str = None) -> str:
        """Get an optional environment variable with a fallback default value.
        
        This method retrieves an environment variable that has a sensible default
        value. It's useful for configuration parameters that have reasonable
        defaults but can be overridden when needed.
        
        Args:
            key: The name of the environment variable to retrieve.
            default: The default value to return if the environment variable
                    is not set. Defaults to None.
            
        Returns:
            str: The value of the environment variable, or the default value
                 if the variable is not set.
        
        """
        EnvironmentLoader.load_environment()
        return os.getenv(key, default)


# Load environment variables immediately when this module is imported
EnvironmentLoader.load_environment()