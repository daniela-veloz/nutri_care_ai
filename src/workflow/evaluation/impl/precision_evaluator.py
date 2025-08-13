from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.workflow.agent_state import AgentState
from src.llm_client import LLMClient
from src.workflow.evaluation.evaluator import Evaluator

class PrecisionEvaluator(Evaluator):
    """Evaluates how precisely responses address user queries in nutrition care contexts.
    
    This evaluator assesses the alignment between what users ask and what the
    system provides in response. It measures precision by analyzing whether
    responses directly address query components, maintain appropriate scope,
    and avoid including irrelevant or tangential information.
    
    The precision evaluation is crucial for ensuring that nutrition care responses
    are focused, relevant, and directly helpful to users. It helps identify when
    responses are too broad, too narrow, or miss key aspects of the original query.
    
    The evaluator uses sophisticated analysis to:
    - Identify core information requests in queries
    - Assess coverage of all query components
    - Evaluate appropriateness of response depth and breadth
    - Detect irrelevant or tangential content
    - Calculate numerical precision scores from 0 to 10
    
    Higher precision scores indicate responses that directly and comprehensively
    address the query without unnecessary information, while lower scores suggest
    the need for query refinement or improved response targeting.
    
    Attributes:
        system_message (str): Detailed instructions for precision evaluation.
        precision_prompt (ChatPromptTemplate): Template for evaluation prompts.
        chain: Processing pipeline for generating precision scores.
    """
    
    def __init__(self):
        self.system_message = """
          You are a specialized evaluation system designed to assess how precisely a
          response addresses a given query.

          Your task is to analyze the alignment between what was asked and what was
          answered, then calculate a numerical precision score


          #Evaluation Process
          For each evaluation request, you will:

          1. Analyze the original query to identify:
            - Core information request
            - Any explicit or implicit constraints
            - Required level of detail
            - Number of distinct questions or components
          2. Examine the response to assess:
            - Direct answers to each component of the query
            - Relevance of all included information
            - Appropriate depth and breadth of information
            - Absence of tangential or unrelated content
          3. Calculate a precision score on a 0-10 scale. Provide only the score, nothing else.

        """

        self.precision_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("user", "Query: {query}\nResponse: {response}\n\nPrecision score:")
        ])

        self.chain = self.precision_prompt | LLMClient.get_llm_client() | StrOutputParser()

    def evaluate(self, state: AgentState) -> AgentState:
        """Evaluate how precisely a response addresses the user's query.
        
        This method analyzes the alignment between the user's query and the
        generated response to determine precision. It assesses whether the
        response directly addresses all components of the query, maintains
        appropriate scope, and avoids including irrelevant information.
        
        The evaluation process includes:
        1. Analyzing the query to identify core information requests and constraints
        2. Examining the response for direct answers to each query component
        3. Assessing the relevance and appropriateness of all included information
        4. Calculating a numerical precision score from 0 to 10
        5. Updating the iteration count for workflow control
        
        Higher scores indicate responses that precisely address the query without
        extraneous information, while lower scores suggest the need for query
        refinement to improve response targeting.
        
        Args:
            state (AgentState): The current workflow state containing:
                               - 'query': The user's original question or request
                               - 'response': The generated response to evaluate
                               - 'precision_loop_count': Current iteration count
        
        Returns:
            AgentState: The updated state with:
                       - 'precision_score': Numerical score (0-10) indicating
                         how precisely the response addresses the query
                       - 'precision_loop_count': Incremented iteration count
        
        Raises:
            KeyError: If required state keys are missing.
            ValueError: If the LLM output cannot be parsed as a float.
            Exception: If there are issues with the evaluation process.
        
        """
        precision_score = float(self.chain.invoke({
            "query": state['query'],
            "response": state['response']
        }))
        
        state['precision_score'] = precision_score
        state['precision_loop_count'] += 1
        
        return state
