from abc import ABC, abstractmethod
from src.workflow.agent_state import AgentState


class Evaluator(ABC):
    """Abstract base class for all response and query evaluators in the RAG workflow.
    
    This abstract class defines the interface for evaluation components in the
    nutrition care AI system. Evaluators are responsible for assessing different
    aspects of the RAG workflow quality, including response groundedness,
    precision, and other quality metrics.
    
    Evaluators play a critical role in the iterative refinement process by
    providing quantitative assessments that drive conditional workflow routing.
    They enable the system to automatically improve response quality through
    feedback loops and iterative processing.
    
    The abstract design allows for flexible implementation of various evaluation
    strategies while maintaining consistent integration with the workflow state
    management system. Each concrete evaluator implementation focuses on specific
    quality aspects relevant to nutrition care applications.
    
    This class follows the Abstract Base Class pattern to ensure all evaluator
    implementations provide the required interface for workflow integration.
    """
    
    @abstractmethod
    def evaluate(self, state: AgentState) -> AgentState:
        """Evaluate workflow state and return updated state with evaluation results.
        
        This abstract method must be implemented by all concrete evaluator classes.
        It defines the core evaluation interface that allows different types of
        evaluators to be seamlessly integrated into the workflow graph.
        
        The method should analyze relevant aspects of the current workflow state,
        compute appropriate quality metrics or scores, and update the state with
        evaluation results. These results are then used by workflow handlers to
        make routing decisions for iterative refinement.
        
        Args:
            state (AgentState): The current workflow state containing all relevant
                               information for evaluation (e.g., query, response,
                               context, previous evaluation scores, iteration counts).
        
        Returns:
            AgentState: The updated state with new evaluation results. The returned
                       state must include any computed scores, metrics, or feedback
                       that will guide subsequent workflow decisions.
        
        Raises:
            NotImplementedError: This method must be implemented by concrete subclasses.
        
        Note:
            Implementations should ensure they properly handle all required state
            keys and provide meaningful error handling for edge cases specific
            to their evaluation domain.
        """
        pass