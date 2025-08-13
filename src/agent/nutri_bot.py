from datetime import datetime
from typing import Dict, List

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.agent.tools.agentic_rag import agentic_rag
from src.llm_client import LLMClient
from src.agent.mem0_client import Mem0Client


class NutritionBot:
    def __init__(self):
        """
        Initialize the NutritionBot class, setting up memory, the LLM client, tools, and the agent executor.
        """

        self.memory = Mem0Client.get_mem0_client()
        self.client = LLMClient.get_llm_client()
        tools = [agentic_rag]

        # Define the system prompt to set the behavior of the chatbot
        system_prompt = """
        You are a caring and knowledgeable Medical Support Agent, specializing in nutrition disorder-related guidance. 
        Your goal is to provide accurate, empathetic, and tailored nutritional recommendations while ensuring a seamless customer experience.
        
        Guidelines for Interaction:
        - Maintain a polite, professional, and reassuring tone.
        - Show genuine empathy for customer concerns and health challenges.
        - Reference past interactions to provide personalized and consistent advice.
        - Engage with the customer by asking about their food preferences, dietary restrictions, and lifestyle before offering recommendations.
        - Ensure consistent and accurate information across conversations.
        - If any detail is unclear or missing, proactively ask for clarification.
        - Always use the agentic_rag tool to retrieve up-to-date and evidence-based nutrition insights.
        - Keep track of ongoing issues and follow-ups to ensure continuity in support.
        - Your primary goal is to help customers make informed nutrition decisions that align with their health conditions and personal preferences.
        """

        # Build the prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),  # System instructions
            ("human", "{input}"),  # Placeholder for human input
            ("placeholder", "{agent_scratchpad}")  # Placeholder for intermediate reasoning steps
        ])

        # Create an agent capable of interacting with tools and executing tasks
        agent = create_tool_calling_agent(self.client, tools, prompt)

        # Wrap the agent in an executor to manage tool interactions and execution flow
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        """
        Store customer interaction in memory for future reference.

        Args:
            user_id (str): Unique identifier for the customer.
            message (str): Customer's query or message.
            response (str): Chatbot's response.
            metadata (Dict, optional): Additional metadata for the interaction.
        """
        if metadata is None:
            metadata = {}

        # Add a timestamp to the metadata for tracking purposes
        metadata["timestamp"] = datetime.now().isoformat()

        # Format the conversation for storage
        conversation = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]

        # Store the interaction in the memory client
        self.memory.add(
            conversation,
            user_id=user_id,
            output_format="v1.1",
            metadata=metadata
        )


    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """
        Retrieve past interactions relevant to the current query.

        Args:
            user_id (str): Unique identifier for the customer.
            query (str): The customer's current query.

        Returns:
            List[Dict]: A list of relevant past interactions.
        """
        return self.memory.search(
            query=query,  # Search for interactions related to the query
            user_id=user_id,  # Restrict search to the specific user
            limit=3  # Complete the code to define the limit for retrieved interactions
        )


    def handle_customer_query(self, user_id: str, query: str) -> str:
        """
        Process a customer's query and provide a response, taking into account past interactions.

        Args:
            user_id (str): Unique identifier for the customer.
            query (str): Customer's query.

        Returns:
            str: Chatbot's response.
        """

        # Retrieve relevant past interactions for context
        relevant_history = self.get_relevant_history(user_id, query)

        # Add check for relevant history format and content
        if not relevant_history or not all(isinstance(memory, dict) and "memory" in memory for memory in relevant_history):
            print("Warning: Relevant history is empty or in an unexpected format.")
            context = ""  # Set empty context if history is invalid
        else:
            # Build a context string from the relevant history
            context = "Previous relevant interactions:\n"
            for memory in relevant_history:
                context += f"Customer: {memory['memory']}\n"  # Customer's past messages
                context += f"Support: {memory['memory']}\n"  # Chatbot's past responses
                context += "---\n"

        # Print context for debugging purposes
        print("Context: ", context)

        # Prepare a prompt combining past context and the current query
        prompt = f"""
        Context:
        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """

        # Generate a response using the agent
        response = self.agent_executor.invoke({"input": prompt})

        # Store the current interaction for future reference
        self.store_customer_interaction(
            user_id=user_id,
            message=query,
            response=response["output"],
            metadata={"type": "support_query"}
        )

        # Return the chatbot's response
        return response['output']