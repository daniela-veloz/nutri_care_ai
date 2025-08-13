from langchain_core.tools import tool


@tool
def agentic_rag(workflow, query: str):
    """
    Runs the RAG-based agent with conversation history for context-aware responses.

    Args:
        query (str): The current user query.

    Returns:
        Dict[str, Any]: The updated state with the generated response and conversation history.
        :param query:
        :param workflow:
    """
    # Initialize state with necessary parameters
    inputs = {
        "query": query,
        "expanded_query": "",
        "context": [],
        "response": "",
        "precision_score": 0,
        "groundedness_score": 0,
        "groundedness_loop_count": 0,
        "precision_loop_count": 0,
        "feedback": "",
        "query_feedback": "",
        "groundedness_check": False,
        "loop_max_iter": 3
    }

    output = workflow.invoke(inputs)

    return output