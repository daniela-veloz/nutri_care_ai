from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.workflow.agent_state import AgentState
from src.llm_client import LLMClient


"""
Improves the expanded query by identifying missing details, specific keywords, or scope refinements that can 
enhance search precision. It does not replace the expanded query but provides structured suggestions for improvement.
"""
class QueryRefiner:
    def __init__(self):
        self.system_message = """
          You are a Query Enhancement Specialist with expertise in search optimization
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

    def refine_query(self, state: AgentState) -> AgentState:
        """
        Suggests improvements for the expanded query.

        Args:
            state (AgentState): The current state of the workflow, containing the query and expanded query.

        Returns:
            AgentState: The updated state with query refinement suggestions.
        """
        suggestions = self.chain.invoke({
            'query': state.query,
            'expanded_query': state.expanded_query
        })
        
        query_feedback = f"Previous Expanded Query: {state.expanded_query}\nSuggestions: {suggestions}"
        state.query_feedback = query_feedback
        
        return state
