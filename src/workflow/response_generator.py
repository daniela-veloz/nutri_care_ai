from langchain.prompts import ChatPromptTemplate
from src.workflow.agent_state import AgentState
from src.llm_client import LLMClient


class ResponseGenerator:
    """Generates contextual responses for nutrition care queries using retrieved documents.
    
    This class is responsible for creating comprehensive, evidence-based responses
    to nutrition-related questions using relevant context retrieved from the vector
    database. It employs sophisticated prompt engineering to ensure responses are
    accurate, well-cited, and grounded in the provided nutritional literature.
    
    The response generator follows strict guidelines for source attribution,
    maintains objectivity, and clearly distinguishes between what can and cannot
    be answered based on the available context. This approach ensures reliable
    and trustworthy nutrition guidance.
    
    Key features include:
    - Source citation with document names and page numbers
    - Clear acknowledgment of information limitations
    - Professional, objective tone suitable for healthcare contexts
    - Structured responses for complex nutritional topics
    - Integration of feedback for response refinement
    
    Attributes:
        system_message (str): Comprehensive instructions for response generation.
        response_prompt (ChatPromptTemplate): Template for contextual response creation.
        chain: Processing pipeline for generating responses from context.
    """
    
    def __init__(self):
        self.system_message = """
          You are an astute assistant specialized in answering questions based on provided context.

          When responding to queries:
          1. Always base your answers solely on the provided context
          2. Cite your sources by including both the source name and page number in parentheses (e.g., "According to the data (Source A, p.23)...")
          3. When quoting directly, use quotation marks and provide the citation
          4. If multiple sources contain relevant information, cite all of them
          5. If the information in the context is insufficient to answer the query completely, clearly state which aspects you can address and which you cannot
          6. If the query cannot be answered at all using the provided context, respond with: "I don't have sufficient information in the provided context to answer this question."
          7. Maintain a professional, objective tone while providing comprehensive answers
          8. Organize complex answers with appropriate structure (paragraphs, bullet points) for clarity
          9. If feedback is provided, use it to refine your response accordingly
          10. Avoid making assumptions or introducing information not present in the context

          Remember that accuracy and proper attribution are paramount. Your role is to effectively retrieve and present information from the provided context, not to generate answers from general knowledge.
        """

        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("user", "Query: {query}\nContext: {context}\n\nfeedback: {feedback}")
        ])

        self.chain = self.response_prompt | LLMClient.get_llm_client()

    def generate_response(self, state: AgentState) -> AgentState:
        """Generate a comprehensive, evidence-based response using retrieved context.
        
        This method creates detailed responses to nutrition-related queries by
        synthesizing information from the retrieved documents. It ensures all
        responses are properly grounded in the provided context, include appropriate
        source citations, and maintain professional standards suitable for healthcare
        applications.
        
        The method processes the query, context documents, and any feedback to
        create responses that are accurate, comprehensive, and properly attributed.
        It follows strict guidelines to avoid generating information not present
        in the context and clearly communicates any limitations in the available
        information.
        
        Args:
            state (AgentState): The current workflow state containing:
                               - 'query': The user's nutrition-related question
                               - 'context': List of retrieved documents with content and metadata
                               - 'feedback': Optional guidance for response refinement
        
        Returns:
            AgentState: The updated state with a 'response' key containing the
                       generated answer. The response includes proper citations,
                       structured formatting, and clear acknowledgment of any
                       information limitations.
        
        Raises:
            KeyError: If required keys ('query', 'context', 'feedback') are missing
                     from the state dictionary.
            Exception: If there are issues with LLM processing or response generation.
        
        """
        response = self.chain.invoke({
            "query": state['query'],
            "context": "\n".join([doc["content"] for doc in state['context']]),
            "feedback": state['feedback']
        })
        
        state['response'] = response.content if hasattr(response, 'content') else str(response)

        return state