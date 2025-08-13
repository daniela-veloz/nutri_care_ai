"""
Agentic RAG Tool

RAG-based agentic tool for processing nutrition care queries through the
comprehensive workflow with context-aware response generation.
"""

from typing import Dict, Any
from langchain_core.tools import tool
from src.workflow.workflow import WorkflowBuilder
from src.workflow.agent_state import AgentState


class AgenticRAG:
    """RAG-based agentic tool for processing nutrition care queries.
    
    This class provides a tool interface for executing the complete RAG workflow
    for nutrition care queries. It integrates with LangChain's tool system to
    enable seamless integration with agentic-based conversational systems.
    
    The class encapsulates the workflow initialization and execution, providing
    a clean interface for processing user queries through the comprehensive
    RAG pipeline including query expansion, context retrieval, response generation,
    and quality evaluation with iterative refinement.
    
    The tool is designed to work within the nutrition care AI system's agentic
    framework, providing context-aware responses based on retrieved nutritional
    information and guidelines.
    """
    
    def __init__(self):
        """Initialize the AgenticRAG tool with workflow builder.
        
        Creates and configures the complete RAG workflow for nutrition care
        query processing. The workflow includes all components needed for
        high-quality response generation with iterative refinement.
        """
        self.workflow = WorkflowBuilder().create_workflow()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a nutrition care query through the complete RAG workflow.
        
        This method executes the full RAG pipeline for a given user query,
        including query expansion, context retrieval, response generation,
        and quality evaluation with iterative refinement as needed.
        
        The method initializes the workflow state with the user query and
        default parameters, then invokes the complete workflow to generate
        a comprehensive, grounded, and precise response.
        
        Args:
            query: The user's nutrition care query to be processed.
            
        Returns:
            Dict[str, Any]: The complete workflow state containing:
                          - 'query': Original user query
                          - 'expanded_query': Enhanced query for better retrieval
                          - 'context': Retrieved relevant documents and metadata
                          - 'response': Final generated response
                          - 'precision_score': Quality score for response precision
                          - 'groundedness_score': Quality score for factual grounding
                          - Other workflow state information
                          
        Raises:
            Exception: If there are issues during workflow execution, such as
                      API failures, retrieval problems, or evaluation errors.
        
        Note:
            The method uses default parameters for loop limits and scoring
            thresholds. These can be customized by modifying the initial
            state configuration as needed.
        """
        # Initialize state with necessary parameters for the RAG workflow
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
        return self.workflow.invoke(initial_state)


# Create a singleton instance for tool usage
_agentic_rag_instance = AgenticRAG()


@tool
def agentic_rag(query: str) -> Dict[str, Any]:
    """
    RAG-based agentic tool for processing nutrition care queries with context-aware responses.

    This tool executes the complete RAG workflow for nutrition care queries,
    including query expansion, context retrieval, response generation, and
    quality evaluation with iterative refinement. It provides high-quality,
    grounded responses based on retrieved nutritional information.

    Args:
        query: The user's nutrition care query to be processed.

    Returns:
        Dict[str, Any]: Complete workflow results including the generated response,
                       quality scores, retrieved context, and other workflow information.
    """
    return _agentic_rag_instance.process_query(query)