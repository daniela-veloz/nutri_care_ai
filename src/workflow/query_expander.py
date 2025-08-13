from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.workflow.agent_state import AgentState
from src.llm_client import LLMClient


class QueryExpander:
    """Expands user queries to improve retrieval accuracy in nutrition care applications.
    
    This class implements query expansion techniques specifically designed for nutrition
    and dietary health domains. It transforms user queries into professional medical
    terminology that enhances semantic search and retrieval from nutritional databases.
    
    The query expander uses a specialized prompt template that incorporates clinical
    terminology, medical concepts, and established nutritional frameworks to create
    multiple reformulated versions of the original query. This approach significantly
    improves the retrieval of relevant nutrition-related documents.
    
    The expansion process follows evidence-based query reformulation strategies that
    are particularly effective for medical and nutritional information retrieval.
    
    Attributes:
        system_message (str): Template for guiding query expansion with clinical context.
        expand_prompt (ChatPromptTemplate): LangChain prompt template for query transformation.
        chain: Processing pipeline combining prompt, LLM, and output parser.
    """
    
    def __init__(self):
        self.system_message = """
          As a nutrition specialist, transform the user query into professional medical terminology that would be used by physicians and dietitians.

          Guidelines for query transformation:
          1. Incorporate precise clinical terminology related to nutritional science, metabolism, and dietary pathophysiology
          2. Expand the original query to include relevant medical concepts and established nutritional frameworks
          3. When appropriate, reference specific assessment tools, biomarkers, or diagnostic criteria used in clinical nutrition

          Query output requirements:
          - Provide exactly 3 reformulated versions of the query unless the original contains multiple distinct questions
          - If the original query contains multiple distinct questions, separate them into individual queries (this may result in more than 3 total queries)
          - Maintain all technical terms, medical acronyms, and specialized vocabulary from the original query without alteration
          - Present only the numbered list of reformulated queries with no introduction or conclusion
          - Ensure each reformulation adds clinical value while preserving the original intent

        """
        
        self.expand_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("user", "Query: {query}")
        ])
        
        self.chain = self.expand_prompt | LLMClient.get_llm_client() | StrOutputParser()

    def expand_query(self, state: AgentState) -> AgentState:
        """Expand a user query into clinical nutrition terminology for improved retrieval.
        
        This method transforms the original user query into professional medical
        terminology commonly used by physicians, dietitians, and nutrition specialists.
        The expansion process creates multiple reformulated versions that incorporate
        precise clinical terminology, relevant medical concepts, and established
        nutritional frameworks.
        
        The method generates exactly 3 reformulated versions of the query (or more
        if the original contains multiple distinct questions), ensuring comprehensive
        coverage of the query intent while maintaining all technical terms and
        medical acronyms from the original query.
        
        Args:
            state (AgentState): The current workflow state containing the user's
                               original query. Must include a 'query' key with
                               the text to be expanded.
        
        Returns:
            AgentState: The updated state with an 'expanded_query' key containing
                       the clinically enhanced query formulations ready for
                       semantic search and retrieval.
        
        Raises:
            KeyError: If the state does not contain the required 'query' key.
            Exception: If there are issues with the LLM processing or prompt execution.
        
        """
        state['expanded_query'] = self.chain.invoke({"query": state['query']})
        return state