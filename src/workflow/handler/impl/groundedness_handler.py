from src.workflow.agent_state import AgentState
from src.workflow.handler.handler import NextStep

class GroundednessNextStep(NextStep):
    """Handles workflow routing decisions based on response groundedness evaluation.
    
    This handler implements conditional routing logic for the groundedness evaluation
    stage of the RAG workflow. It determines whether generated responses are
    sufficiently grounded in retrieved context or require response refinement
    to improve factual accuracy and reduce hallucinations.
    
    Groundedness evaluation is critical for ensuring that nutrition care responses
    are based on reliable source material rather than model-generated content
    that may be inaccurate or misleading. The handler enforces quality standards
    by routing poorly grounded responses back for refinement.
    
    The handler balances the need for well-grounded responses with practical
    constraints on processing iterations, preventing infinite refinement loops
    while allowing reasonable attempts to improve response quality.
    
    Attributes:
        groundedness_threshold (float): Minimum groundedness score (0-10) required
                                      for responses to proceed to precision checking.
    """
    
    def __init__(self, groundedness_threshold: float = 8.0):
        """Initialize the groundedness handler with quality threshold.
        
        Args:
            groundedness_threshold: Minimum groundedness score (0-10) required
                                  for responses to be considered sufficiently
                                  grounded in context. Higher values enforce
                                  stricter adherence to source material.
        """
        self.groundedness_threshold = groundedness_threshold

    def get_next_step(self, state: AgentState) -> str:
        """Determine next workflow step based on groundedness evaluation results.
        
        This method analyzes the groundedness score and iteration count to make
        routing decisions for the workflow. It implements a three-way decision logic:
        
        1. If groundedness meets threshold: Proceed to precision check ("check_precision")
        2. If groundedness is low but iterations remain: Refine response ("refine_response")
        3. If max iterations reached: Terminate workflow ("max_iterations_reached")
        
        The routing ensures that only well-grounded responses proceed to the next
        evaluation stage, maintaining high standards for factual accuracy in
        nutrition care applications.
        
        Args:
            state (AgentState): Current workflow state containing:
                               - 'groundedness_score': Current groundedness evaluation (0-10)
                               - 'groundedness_loop_count': Number of groundedness refinement attempts
                               - 'loop_max_iter': Maximum allowed iterations for refinement
        
        Returns:
            str: Next workflow step identifier:
                 - "check_precision": Groundedness is satisfactory, proceed to precision check
                 - "refine_response": Groundedness needs improvement, refine response
                 - "max_iterations_reached": Iteration limit exceeded, terminate
        
        """
        if state['groundedness_score'] >= self.groundedness_threshold:
            return "check_precision"
        else:
            if state['groundedness_loop_count'] > state['loop_max_iter']:
                return "max_iterations_reached"
            else:
                return "refine_response"