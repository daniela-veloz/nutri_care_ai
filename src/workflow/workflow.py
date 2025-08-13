from langgraph.graph import END, StateGraph, START
from src.workflow.agent_state import AgentState
from src.workflow.context_retriever import ContextRetriever
from src.workflow.evaluation.impl.groundedness_evaluator import GroundednessEvaluator
from src.workflow.evaluation.impl.precision_evaluator import PrecisionEvaluator
from src.workflow.feedback.impl.query_feedback_provider import QueryFeedbackProvider
from src.workflow.feedback.impl.response_feedback_provider import ResponseFeedbackProvider
from src.workflow.handler.impl.max_iterations_handler import MaxIterationsHandler
from src.workflow.handler.impl.precision_handler import PrecisionHandler
from src.workflow.handler.impl.groundedness_handler import GroundednessNextStep
from src.workflow.query_expander import QueryExpander
from src.workflow.response_generator import ResponseGenerator


class WorkflowBuilder:
    """Orchestrates the complete RAG workflow for nutrition care AI applications.
    
    This class constructs and manages a sophisticated Retrieval-Augmented Generation
    (RAG) workflow specifically designed for nutrition and dietary health applications.
    It coordinates multiple components including query expansion, context retrieval,
    response generation, evaluation, and iterative refinement.
    
    The workflow implements a state graph architecture using LangGraph, enabling
    complex conditional logic and iterative processing. It includes quality control
    mechanisms such as groundedness evaluation and precision checking to ensure
    high-quality, reliable responses.
    
    Key workflow components:
    - Query expansion for improved retrieval
    - Semantic context retrieval from nutrition databases
    - Evidence-based response generation
    - Groundedness evaluation for factual accuracy
    - Precision evaluation for query-response alignment
    - Iterative refinement with feedback loops
    - Maximum iteration handling for workflow termination
    
    The workflow supports both single-pass and iterative processing, automatically
    refining queries and responses based on evaluation metrics until satisfactory
    quality is achieved or maximum iterations are reached.
    
    Attributes:
        workflow (StateGraph): The main workflow graph managing state transitions.
        query_expander (QueryExpander): Component for expanding user queries.
        context_retriever (ContextRetriever): Component for retrieving relevant documents.
        response_generator (ResponseGenerator): Component for generating responses.
        groundedness_evaluator (GroundednessEvaluator): Evaluates response grounding.
        response_feedback_provider (ResponseFeedbackProvider): Provides response feedback.
        precision_evaluator (PrecisionEvaluator): Evaluates response precision.
        query_feedback_provider (QueryFeedbackProvider): Provides query feedback.
        max_iterations_handler (MaxIterationsHandler): Handles iteration limits.
        precision_handler (PrecisionHandler): Manages precision-based routing.
        groundedness_next_step (GroundednessNextStep): Manages groundedness-based routing.
    """
    
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self.query_expander = QueryExpander()
        self.context_retriever = ContextRetriever()
        self.response_generator = ResponseGenerator()
        self.groundedness_evaluator = GroundednessEvaluator()
        self.response_feedback_provider = ResponseFeedbackProvider()
        self.precision_evaluator = PrecisionEvaluator()
        self.query_feedback_provider = QueryFeedbackProvider()
        self.max_iterations_handler = MaxIterationsHandler()
        self.precision_handler = PrecisionHandler()
        self.groundedness_next_step = GroundednessNextStep()


    def create_workflow(self) -> StateGraph:
        """Create and configure the complete RAG workflow for nutrition care AI.
        
        This method constructs a sophisticated state graph workflow that implements
        a complete RAG (Retrieval-Augmented Generation) pipeline with quality control
        and iterative refinement capabilities. The workflow is specifically optimized
        for nutrition and dietary health applications.
        
        The workflow follows this general pattern:
        1. Query expansion to improve retrieval effectiveness
        2. Context retrieval from nutrition document database
        3. Response generation using retrieved context
        4. Groundedness evaluation to ensure factual accuracy
        5. Response refinement if needed (iterative)
        6. Precision evaluation to ensure query-response alignment
        7. Query refinement if needed (iterative)
        8. Termination when quality thresholds are met or max iterations reached
        
        The workflow includes conditional branching based on evaluation results,
        allowing for automatic quality improvement through iterative refinement.
        This ensures that responses meet high standards for accuracy and relevance
        in nutrition care applications.
        
        Returns:
            StateGraph: A fully configured workflow graph ready for compilation and
                       execution. The graph includes all nodes, edges, and conditional
                       routing logic necessary for complete RAG processing.
        
        Raises:
            Exception: If there are issues configuring workflow components or
                      setting up the state graph structure.
        
        Note:
            The returned StateGraph must be compiled before use. The workflow
            automatically handles state management and routing between components
            based on evaluation results and iteration counts.
        """
        # Add processing nodes
        self.workflow.add_node("expand_query", self.query_expander.expand_query)         # Step 1: Expand user query.
        self.workflow.add_node("retrieve_context", self.context_retriever.retrieve_context)     # Step 2: Retrieve relevant documents.
        self.workflow.add_node("generate_response", self.response_generator.generate_response)       # Step 3: Generate a response based on retrieved data.
        self.workflow.add_node("score_groundedness", self.groundedness_evaluator.evaluate)   # Step 4: Evaluate response grounding.
        self.workflow.add_node("refine_response", self.response_feedback_provider.provide_feedback)      # Step 5: Improve response if it's weakly grounded.
        self.workflow.add_node("check_precision", self.precision_evaluator.evaluate)      # Step 6: Evaluate response precision.
        self.workflow.add_node("refine_query", self.query_feedback_provider.provide_feedback)         # Step 7: Improve query if response lacks precision.
        self.workflow.add_node("max_iterations_reached", self.max_iterations_handler.handle_max_iterations)  # Step 8: Handle max iterations.

        # Main flow edges
        self.workflow.add_edge(START, "expand_query")
        self.workflow.add_edge("expand_query", "retrieve_context")
        self.workflow.add_edge("retrieve_context", "generate_response")
        self.workflow.add_edge("generate_response", "score_groundedness")

        # Conditional edges based on groundedness check
        self.workflow.add_conditional_edges(
            "score_groundedness",
            self.groundedness_next_step.get_next_step,  # Use the conditional function
            {
                "check_precision": "check_precision",  # If well-grounded, proceed to precision check.
                "refine_response": "refine_response",  # If not, refine the response.
                "max_iterations_reached": "max_iterations_reached"  # If max loops reached, exit.
            }
        )

        self.workflow.add_edge("refine_response", "generate_response")  # Refined responses are reprocessed.

        # Conditional edges based on precision check
        self.workflow.add_conditional_edges(
            "check_precision",
            self.precision_handler.get_next_step,  # Use the conditional function
            {
                "pass": END,              # If precise, complete the workflow.
                "refine_query": "refine_query",  # If imprecise, refine the query.
                "max_iterations_reached": "max_iterations_reached"  # If max loops reached, exit.
            }
        )

        self.workflow.add_edge("refine_query", "expand_query")  # Refined queries go through expansion again.
        self.workflow.add_edge("max_iterations_reached", END)

        return self.workflow
