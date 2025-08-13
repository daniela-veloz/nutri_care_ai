from src.workflow.agent_state import AgentState


class MaxIterationsHandler:
    """Handles workflow termination when maximum iteration limits are reached.
    
    This handler provides graceful termination of the RAG workflow when the
    iterative refinement process has reached its maximum allowed iterations
    without achieving the desired quality thresholds. It ensures the system
    provides a meaningful response even when optimal quality cannot be achieved
    within the iteration constraints.
    
    The handler is essential for preventing infinite loops in the refinement
    process while maintaining system responsiveness. It provides users with
    a helpful fallback message that acknowledges limitations rather than
    failing silently or continuing indefinitely.
    
    The fallback approach maintains transparency about the system's limitations
    and guides users toward potential solutions, such as providing more specific
    queries or additional context for better results.
    
    Attributes:
        fallback_message (str): The message to display when max iterations are
                               reached without achieving quality thresholds.
    """
    
    def __init__(self, fallback_message: str = "We need more context to provide an accurate answer."):
        """Initialize the max iterations handler with a fallback message.
        
        Args:
            fallback_message: The message to return when maximum iterations
                            are reached. Should be helpful and acknowledge
                            the limitation while guiding the user.
        """
        self.fallback_message = fallback_message

    def handle_max_iterations(self, state: AgentState) -> AgentState:
        """Provide a fallback response when maximum iterations are reached.
        
        This method is called when the workflow has reached its maximum allowed
        iterations for refinement without achieving the desired quality thresholds.
        It replaces the current response with a helpful fallback message that
        acknowledges the limitation and provides guidance to the user.
        
        The method ensures graceful termination of the iterative process while
        maintaining a positive user experience by providing transparency about
        the system's limitations and suggesting potential improvements.
        
        Args:
            state (AgentState): The current workflow state that has reached
                               maximum iterations. The state will be updated
                               with the fallback response.
        
        Returns:
            AgentState: The updated state with the fallback message as the
                       final response. This terminates the workflow processing.
        
        """
        state['response'] = self.fallback_message
        return state