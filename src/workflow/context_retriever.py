from langchain_community.vectorstores import Chroma

from src.workflow.agent_state import AgentState
from src.workflow.embedding_model import EmbeddingModel


class ContextRetriever:
    """Retrieves relevant context from the vector database for RAG operations.
    
    This class is responsible for finding and retrieving the most relevant
    nutrition-related documents from the ChromaDB vector store based on
    semantic similarity to user queries. It serves as a critical component
    in the RAG (Retrieval-Augmented Generation) pipeline by providing
    contextually relevant information to enhance response generation.
    
    The retriever uses OpenAI embeddings to perform semantic search,
    finding documents that are conceptually similar to the query rather
    than just matching keywords. This enables more accurate and relevant
    context retrieval for nutrition care questions.
    
    Attributes:
        retriever: A LangChain retriever configured for similarity search
                  against the nutritional document database.
    """
    
    def __init__(self):
        """Initialize the context retriever with ChromaDB vector store.
        
        Sets up a connection to the persistent ChromaDB vector store containing
        nutrition-related documents. The vector store is configured with:
        - Collection name: 'semantic_chunks' for organized document storage
        - Persist directory: './nutritional_db' for local storage
        - Embedding function: OpenAI embeddings for semantic similarity
        - Search type: Similarity search for finding relevant documents
        - Search parameters: Top 3 most relevant documents (k=3)
        
        Raises:
            Exception: If the vector store cannot be initialized or if the
                      nutritional_db directory is not accessible.
        """
        # Initialize the Chroma vector store for retrieving documents
        vector_store = Chroma(
            collection_name='semantic_chunks',
            persist_directory="./nutritional_db",
            embedding_function=EmbeddingModel.get_embedding_model()
        )

        # Create a retriever from the vector store
        self.retriever = vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 3}
        )

    def retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant context documents for the given query.
        
        This method performs semantic search against the nutrition document
        database to find the most relevant context for answering the user's
        query. It uses the expanded query (if available) or falls back to
        the original query to ensure optimal retrieval performance.
        
        The retrieved documents are formatted with both content and metadata
        to provide comprehensive context for response generation. Each document
        includes the full text content and associated metadata such as source
        information, document type, and other relevant attributes.
        
        Args:
            state (AgentState): The current workflow state containing the query
                               and expanded query. Must include an 'expanded_query'
                               key with the search text.
        
        Returns:
            AgentState: The updated state with a 'context' key containing a list
                       of retrieved documents. Each document is a dictionary with
                       'content' (text) and 'metadata' (document attributes).
        
        Raises:
            KeyError: If the state does not contain the required 'expanded_query' key.
            Exception: If there are issues with the vector store retrieval process.
        
        """
        query = state['expanded_query']
        docs = self.retriever.invoke(query)
        state['context'] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        return state