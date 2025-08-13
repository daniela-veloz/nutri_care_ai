"""
Guardrail Filter

Input filtering system using Groq's Llama Guard for content safety validation.
This module provides guardrail functionality to ensure user inputs are safe
and appropriate for nutrition care AI processing.
"""

import os
from groq import Groq
from src.environment_loader import EnvironmentLoader


class GuardFilter:
    """Singleton class for managing Groq Llama Guard client for content filtering.
    
    This class provides content safety filtering using Groq's Llama Guard models
    to ensure user inputs are appropriate and safe for processing by the nutrition
    care AI system. It implements safety guardrails to prevent harmful, inappropriate,
    or malicious content from being processed.
    
    The class uses a singleton pattern to efficiently manage the Groq client
    connection and provides lazy initialization to prevent import-time errors
    when API keys are not immediately available.
    
    Content filtering is essential for maintaining the integrity and safety of
    the nutrition care AI system, ensuring that only appropriate queries are
    processed and potentially harmful content is filtered out.
    """
    _guard_client = None
    
    @staticmethod
    def get_guard_client() -> Groq:
        """Get or create a Groq client instance for Llama Guard filtering.
        
        This method implements lazy initialization to create a Groq client
        only when needed. It retrieves the API key from environment variables
        using the centralized EnvironmentLoader, with fallback to direct
        environment access for flexibility.
        
        Returns:
            Groq: A configured Groq client instance for content filtering.
            
        Raises:
            ValueError: If the GROQ_API_KEY environment variable is not set.
        """
        if GuardFilter._guard_client is None:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                api_key = EnvironmentLoader.get_required_env('GROQ_API_KEY')
            
            GuardFilter._guard_client = Groq(api_key=api_key)
        
        return GuardFilter._guard_client
    
    @staticmethod
    def filter(user_input: str, model: str = "meta-llama/llama-guard-4-12b") -> str:
        """Filter user input using Llama Guard to ensure content safety.
        
        This method processes user input through Groq's Llama Guard model to
        identify and filter potentially harmful, inappropriate, or malicious
        content. It serves as a safety layer before processing queries in the
        nutrition care AI workflow.
        
        The filtering process analyzes the input content and returns either the
        filtered safe content or None if the content is deemed unsafe or if
        processing encounters errors.
        
        Args:
            user_input: The raw input provided by the user that needs filtering.
            model: The Llama Guard model to use for filtering. Defaults to
                  "llama-3.1-8b-instant" which provides balanced safety and performance.
        
        Returns:
            str: The filtered and validated input content, or None if the input
                 is unsafe or if processing fails.
        
        Raises:
            Exception: Captures and logs any errors during the filtering process,
                      returning None to indicate filtering failure.
        
        Note:
            This method should be called before processing any user input in
            the nutrition care workflow to ensure content safety and system
            integrity.
        """
        try:
            guard_client = GuardFilter.get_guard_client()
            
            # Create a request to Llama Guard to filter the user input
            response = guard_client.chat.completions.create(
                messages=[{"role": "user", "content": user_input}],
                model=model,
            )
            # Return the filtered input
            return response.choices[0].message.content.strip().replace("\n", " ")
        except Exception as e:
            print(f"Error with Llama Guard: {e}")
            return None