from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.workflow.agent_state import AgentState
from src.llm_client import LLMClient
from src.workflow.feedback.feedback_provider import FeedbackProvider


class QueryFeedbackProvider(FeedbackProvider):
    """Provides feedback to enhance query expansion and search precision in nutrition contexts.
    
    This feedback provider specializes in analyzing both original user queries and
    their expanded versions to suggest improvements for better search effectiveness
    and information retrieval. It acts as a query enhancement expert that identifies
    opportunities to refine search strategies without replacing existing expansions.
    
    The provider focuses on optimizing search precision by analyzing query structure,
    terminology usage, and search strategy effectiveness. It provides targeted
    suggestions to improve the expanded query's ability to retrieve relevant
    nutrition information from the document database.
    
    Key areas of feedback include:
    - Search term optimization for better semantic matching
    - Clinical terminology enhancement for nutrition contexts
    - Query structure improvements for comprehensive coverage
    - Specificity adjustments for precision optimization
    - Alternative search strategies for better retrieval
    
    The feedback is designed to guide query expansion refinement through multiple
    iterations, helping the system progressively improve its ability to find
    relevant nutritional information for user questions.
    
    Attributes:
        system_message (str): Instructions for query enhancement analysis.
        refine_query_prompt (ChatPromptTemplate): Template for generating query feedback.
        chain: Processing pipeline for analyzing queries and generating suggestions.
    """
    
    def __init__(self):
        self.system_message = """
          You are a llm query enhancer with expertise in search optimization
          and information retrieval. Your task is to analyze both an original query
          and its expanded version, then provide targeted suggestions to further
          improve search precision without replacing the expanded query.

          Do not replace the expanded query but provide structured suggestions for improvement.

        """

        self.refine_query_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("user", "Original Query: {query}\nExpanded Query: {expanded_query}\n\n"
                     "What improvements can be made for a better search?")
        ])

        self.chain = self.refine_query_prompt | LLMClient.get_llm_client() | StrOutputParser()

    def provide_feedback(self, state: AgentState) -> AgentState:
        """Analyze query expansion and provide suggestions for improved search precision.
        
        This method examines both the original user query and its expanded version
        to identify opportunities for enhancing search effectiveness. It provides
        targeted suggestions to improve the expanded query's ability to retrieve
        relevant nutrition information without directly replacing the expansion.
        
        The analysis focuses on search optimization aspects including terminology
        usage, query structure, specificity levels, and potential alternative
        search strategies that could yield better retrieval results in nutrition
        care contexts.
        
        Args:
            state (AgentState): The current workflow state containing:
                               - 'query': The original user query
                               - 'expanded_query': The current expanded version
        
        Returns:
            AgentState: The updated state with a 'query_feedback' key containing:
                       - The previous expanded query for reference
                       - Specific suggestions for search improvement
                       - Guidance for better terminology and structure
        
        Raises:
            KeyError: If required state keys ('query', 'expanded_query') are missing.
            Exception: If there are issues with the feedback generation process.
        
        """
        suggestions = self.chain.invoke({
            'query': state['query'],
            'expanded_query': state['expanded_query']
        })
        
        query_feedback = f"Previous Expanded Query: {state['expanded_query']}\nSuggestions: {suggestions}"
        state['query_feedback'] = query_feedback
        
        return state
