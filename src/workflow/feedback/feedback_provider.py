from abc import ABC, abstractmethod
from src.workflow.agent_state import AgentState


class FeedbackProvider(ABC):
    """Abstract base class for feedback providers in the RAG workflow refinement system.
    
    This abstract class defines the interface for components that provide constructive
    feedback to improve various aspects of the RAG workflow. Feedback providers analyze
    current workflow states and generate suggestions for enhancement without directly
    modifying the content they evaluate.
    
    Feedback providers are essential for the iterative refinement process in the
    nutrition care AI system. They enable continuous improvement by identifying
    specific areas where responses or queries can be enhanced for better accuracy,
    completeness, and precision.
    
    The abstract design supports different types of feedback providers, such as:
    - Response feedback for improving generated answers
    - Query feedback for enhancing search effectiveness
    - Context feedback for better document retrieval
    
    Each concrete implementation focuses on specific aspects of quality improvement
    while maintaining consistent integration with the workflow state management.
    """
    
    @abstractmethod
    def provide_feedback(self, state: AgentState) -> AgentState:
        """Analyze workflow state and provide constructive feedback for improvement.
        
        This abstract method must be implemented by all concrete feedback provider
        classes. It defines the core feedback interface that enables different types
        of feedback providers to be integrated into the workflow graph for iterative
        refinement.
        
        The method should analyze relevant aspects of the current workflow state,
        identify opportunities for improvement, and generate constructive feedback
        that can guide subsequent workflow processing. The feedback should be
        specific, actionable, and focused on enhancing quality metrics.
        
        Args:
            state (AgentState): The current workflow state containing all relevant
                               information for feedback generation (e.g., queries,
                               responses, context, evaluation scores, previous feedback).
        
        Returns:
            AgentState: The updated state with new feedback that can be used to
                       guide refinement processes. The feedback should be stored
                       in appropriate state keys for subsequent workflow components.
        
        Raises:
            NotImplementedError: This method must be implemented by concrete subclasses.
        
        Note:
            Implementations should focus on providing constructive suggestions rather
            than directly modifying the content being analyzed. The goal is to guide
            improvement processes, not to replace existing content.
        """
        pass