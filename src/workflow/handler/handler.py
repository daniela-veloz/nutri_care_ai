from abc import ABC, abstractmethod
from src.workflow.agent_state import AgentState


class NextStep(ABC):
    """Abstract base class for workflow routing decision handlers in the RAG system.
    
    This abstract class defines the interface for components that make routing
    decisions in the workflow graph based on current state conditions. These
    handlers implement the conditional logic that enables dynamic workflow
    routing in the nutrition care AI system.
    
    Next step handlers are crucial for implementing intelligent workflow control,
    allowing the system to make decisions about whether to proceed to the next
    stage, initiate refinement cycles, or terminate processing based on quality
    metrics and iteration limits.
    
    The abstract design supports various types of routing logic including:
    - Quality threshold checking (e.g., groundedness, precision scores)
    - Iteration limit management
    - Conditional workflow branching
    - Termination condition handling
    
    Each concrete implementation focuses on specific decision criteria while
    maintaining consistent integration with the LangGraph conditional routing
    system. The returned string values must match the defined workflow routes.
    
    This pattern enables sophisticated workflow orchestration with automatic
    quality control and iterative improvement capabilities essential for
    reliable nutrition care applications.
    """
    
    @abstractmethod
    def get_next_step(self, state: AgentState) -> str:
        """Determine the next workflow step based on current state conditions.
        
        This abstract method must be implemented by all concrete handler classes.
        It defines the routing logic interface that enables conditional workflow
        navigation in the LangGraph state machine. The method analyzes relevant
        state information and returns a route identifier for workflow progression.
        
        The routing decision should be based on appropriate quality metrics,
        iteration counts, and other relevant state conditions specific to the
        handler's purpose. The returned value must correspond to a valid route
        defined in the workflow graph configuration.
        
        Args:
            state (AgentState): The current workflow state containing evaluation
                               scores, iteration counts, quality metrics, and
                               other relevant information for routing decisions.
        
        Returns:
            str: A route identifier that matches the conditional edges defined
                 in the workflow graph. Common values include "pass", "refine_query",
                 "refine_response", "max_iterations_reached", "check_precision", etc.
        
        Raises:
            NotImplementedError: This method must be implemented by concrete subclasses.
        
        Note:
            Implementations should ensure robust handling of edge cases and
            provide appropriate fallback routes when standard conditions are
            not met. The routing logic should align with the overall workflow
            quality objectives.
        """
        pass