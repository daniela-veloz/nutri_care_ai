from src.workflow.agent_state import AgentState
from src.workflow.handler.handler import NextStep


class PrecisionHandler(NextStep):
    """Handles workflow routing decisions based on response precision evaluation.
    
    This handler implements conditional routing logic for the precision evaluation
    stage of the RAG workflow. It determines whether responses adequately address
    user queries with sufficient precision or require query refinement for better
    targeting of nutritional information.
    
    The handler uses a configurable precision threshold to assess response quality
    and manages iteration limits to prevent excessive refinement cycles. It balances
    the need for high-quality responses with practical constraints on processing
    time and computational resources.
    
    Precision evaluation focuses on how well responses address the specific
    information needs expressed in user queries, ensuring that nutrition care
    responses are focused, relevant, and directly helpful rather than overly
    broad or tangential.
    
    Attributes:
        precision_threshold (float): Minimum precision score (0-10) required
                                   for responses to pass evaluation.
        max_loops (int): Maximum number of precision refinement iterations
                        allowed before terminating the workflow.
    """
    
    def __init__(self, precision_threshold: float = 8.0, max_loops: int = 3):
        """Initialize the precision handler with quality and iteration parameters.
        
        Args:
            precision_threshold: Minimum precision score (0-10) required for
                               responses to be considered satisfactory. Higher
                               values demand more precise responses.
            max_loops: Maximum number of precision refinement iterations allowed
                      before triggering termination. Prevents infinite loops
                      while allowing reasonable improvement attempts.
        """
        self.precision_threshold = precision_threshold
        self.max_loops = max_loops

    def get_next_step(self, state: AgentState) -> str:
        """Determine next workflow step based on precision evaluation results.
        
        This method analyzes the current precision score and iteration count to
        make routing decisions for the workflow. It implements a three-way
        decision logic:
        
        1. If precision meets the threshold: Proceed to completion ("pass")
        2. If precision is low but iterations remain: Refine query ("refine_query")
        3. If max iterations reached: Terminate workflow ("max_iterations_reached")
        
        The routing ensures that only high-quality, precisely targeted responses
        are delivered to users while maintaining reasonable processing boundaries.
        
        Args:
            state (AgentState): Current workflow state containing:
                               - 'precision_score': Current precision evaluation (0-10)
                               - 'precision_loop_count': Number of precision refinement attempts
        
        Returns:
            str: Next workflow step identifier:
                 - "pass": Precision is satisfactory, complete workflow
                 - "refine_query": Precision needs improvement, refine query
                 - "max_iterations_reached": Iteration limit exceeded, terminate
        
        """
        if state['precision_score'] >= self.precision_threshold:
            return "pass"
        else:
            if state['precision_loop_count'] > self.max_loops:
                return "max_iterations_reached"
            else:
                return "refine_query"