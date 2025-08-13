from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.workflow.agent_state import AgentState
from src.llm_client import LLMClient
from src.workflow.feedback.feedback_provider import FeedbackProvider


class ResponseFeedbackProvider(FeedbackProvider):
    """Provides constructive feedback to improve generated responses in nutrition care contexts.
    
    This feedback provider analyzes generated responses and identifies specific
    opportunities for enhancement without directly rewriting the content. It focuses
    on improving accuracy, completeness, clarity, and overall quality of nutrition-
    related responses through targeted suggestions.
    
    The provider acts as a specialized data analyst that examines responses in
    relation to their originating queries and provides structured feedback on:
    - Information gaps that should be addressed
    - Ambiguities that need clarification
    - Accuracy concerns requiring attention
    - Completeness issues affecting usefulness
    - Structure and flow improvements for better readability
    
    This feedback is crucial for the iterative refinement process, enabling the
    system to generate increasingly accurate and comprehensive responses to
    nutrition care questions through multiple refinement cycles.
    
    The feedback is formatted to include both the previous response and specific
    suggestions, providing context for the response generator to create improved
    versions while maintaining the original intent and factual grounding.
    
    Attributes:
        system_message (str): Instructions for the feedback analysis process.
        refine_response_prompt (ChatPromptTemplate): Template for generating feedback.
        chain: Processing pipeline for analyzing responses and generating suggestions.
    """
    
    def __init__(self):
        self.system_message = """
          You are a data analyst specializing in constructive feedback
          on information retrieval outputs. Your task is to analyze a generated
          response in relation to its query, then provide targeted suggestions
          for improvement without rewriting the response.

          #Objectives
          Your role is to identify specific opportunities to enhance the response by focusing on:
            - Information Gaps
            - Ambiguities
            - Accuracy Concerns
            - Completeness
            - Structure and Flow
         """

        self.refine_response_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("user", "Query: {query}\nResponse: {response}\n\n"
                     "What improvements can be made to enhance accuracy and completeness?")
        ])

        self.chain = self.refine_response_prompt | LLMClient.get_llm_client() | StrOutputParser()

    def provide_feedback(self, state: AgentState) -> AgentState:
        """Analyze the generated response and provide constructive improvement suggestions.
        
        This method examines the generated response in relation to the original query
        and identifies specific opportunities for enhancement. It provides targeted
        feedback focusing on information gaps, ambiguities, accuracy concerns,
        completeness issues, and structural improvements.
        
        The feedback is designed to guide the response generator in creating improved
        versions while maintaining factual grounding and addressing the user's
        information needs more comprehensively. The analysis considers both the
        content quality and the alignment with nutrition care best practices.
        
        Args:
            state (AgentState): The current workflow state containing:
                               - 'query': The original user query for context
                               - 'response': The generated response to analyze
        
        Returns:
            AgentState: The updated state with a 'feedback' key containing:
                       - The previous response for reference
                       - Specific suggestions for improvement
                       - Guidance for enhancing accuracy and completeness
        
        Raises:
            KeyError: If required state keys ('query', 'response') are missing.
            Exception: If there are issues with the feedback generation process.
        
        """
        suggestions = self.chain.invoke({
            'query': state['query'],
            'response': state['response']
        })
        
        feedback = f"Previous Response: {state['response']}\nSuggestions: {suggestions}"
        state['feedback'] = feedback
        
        return state
