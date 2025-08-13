#!/usr/bin/env python3
"""
Simple Nutrition Care AI Application

This application demonstrates the RAG (Retrieval-Augmented Generation) workflow
for nutrition-related queries using LangChain, ChromaDB, and OpenAI.
"""


from src.agentic.agent.nutri_agent import NutritionAgent


def main():
    """Main application entry point."""
    nutrition_agent = NutritionAgent()
    nutrition_agent.run()

if __name__ == "__main__":
    main()