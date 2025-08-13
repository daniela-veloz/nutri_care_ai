#!/usr/bin/env python3
"""
Simple Nutrition Care AI Application

This application demonstrates the RAG (Retrieval-Augmented Generation) workflow
for nutrition-related queries using LangChain, ChromaDB, and OpenAI.
"""

import os

from IPython.core.display import Image
from IPython.core.display_functions import display

# Disable telemetry to avoid warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ANALYTICS"] = "false"
from src.workflow.agent_state import AgentState
from src.workflow.workflow import WorkflowBuilder
from src.environment_loader import EnvironmentLoader


def main():
    """Main application entry point."""

    # Verify required environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            EnvironmentLoader.get_required_env('OPENAI_API_KEY')
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    print("Welcome to Nutrition Care AI!")
    print("Ask me any nutrition-related questions.\n")
    
    # Initialize workflow
    try:
        workflow = WorkflowBuilder().create_workflow().compile()
        print("Workflow initialized successfully")
    except Exception as e:
        print(f"Failed to initialize workflow: {e}")
        return
    
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
            
            # Initialize state
            initial_state: AgentState = {
                "query": query,
                "expanded_query": "",
                "context": [],
                "response": "",
                "precision_score": 0.0,
                "groundedness_score": 0.0,
                "groundedness_loop_count": 0,
                "precision_loop_count": 0,
                "feedback": "",
                "query_feedback": "",
                "groundedness_check": False,
                "loop_max_iter": 3
            }
            
            print(f"\nProcessing query: '{query}'")
            print("Please wait...")
            
            # Run workflow
            result = workflow.invoke(initial_state)
            
            # Display results
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Original Query: {result['query']}")
            print(f"Expanded Query: {result['expanded_query']}")
            print(f"Groundedness Score: {result['groundedness_score']:.1f}/10")
            print(f"Precision Score: {result['precision_score']:.1f}/10")
            print(f"Refinement Loops: G={result['groundedness_loop_count']}, P={result['precision_loop_count']}")
            print("\nResponse:")
            print("-" * 40)
            print(result['response'])
            print("="*60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()