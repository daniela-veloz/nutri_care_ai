#!/usr/bin/env python3
"""
Simple Nutrition Care AI Application

This application demonstrates the RAG (Retrieval-Augmented Generation) workflow
for nutrition-related queries using LangChain, ChromaDB, and OpenAI.
"""

import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO

# Suppress all warnings
warnings.filterwarnings("ignore")

from IPython.core.display import Image
from IPython.core.display_functions import display
from src.agent.nutri_bot import NutritionBot
from src.environment_loader import EnvironmentLoader


def main():
    """Main application entry point."""
    
    print("Welcome to Nutrition Care AI!")
    print("Ask me any nutrition-related questions.\n")
    
    # Initialize chatbot with stderr suppression to avoid telemetry warnings
    try:
        # Suppress stderr during initialization to hide telemetry warnings
        stderr_buffer = StringIO()
        with redirect_stderr(stderr_buffer):
            chatbot = NutritionBot()  # Initialize chatbot instance
        print("NutritionBot initialized successfully")
    except Exception as e:
        print(f"Failed to initialize NutritionBot: {e}")
        return
    
    # Demo query
    try:
        print("\n" + "="*60)
        print("DEMO QUERY")
        print("="*60)
        response = chatbot.handle_customer_query("Daniela", "what are the most common nutrition disorders?")  # Call chatbot handler function
        print(f"Response: {response}")
        print("="*60)
    except Exception as e:
        print(f"Error with demo query: {e}")
    
    # Interactive query loop
    while True:
        try:
            # Get user input
            query = input("\nEnter your nutrition question (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using Nutrition Care AI!")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            print(f"\nProcessing query: '{query}'")
            print("Please wait...")
            
            # Process query using chatbot
            response = chatbot.handle_customer_query("user", query)
            
            # Display results
            print("\n" + "="*60)
            print("RESPONSE")
            print("="*60)
            print(response)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()