from src.agentic.agent.nutri_bot import NutritionBot
from src.agentic.guardrail.guard_filter import GuardFilter

"""
    A conversational agent that answers nutrition disorder-related questions
    using a RAG-based workflow with safety filtering and user session handling.
    """
class NutritionAgent:

    def __init__(self):
        self.nutrition_bot = NutritionBot()
        self.guard_filter = GuardFilter()

    def run(self):
        print("Welcome to the Nutrition Disorder Specialist Agent!")
        print("You can ask me anything about nutrition disorders, such as symptoms, causes, treatments, and more.")
        print("Type 'exit' to end the conversation.\n")

        print("Login by providing customer name")  # This provides a way to initiate a chat as different users.
        user_id = input()  # Get user ID for tracking conversation sessions

        while True:
            # Get user input
            print("How can I help you?")
            user_query = input("You: ")

            # Define the logic for exitting the loop' [if user types in exit]
            if user_query.lower() == "exit":
                print("Agent: Goodbye! Feel free to return if you have more questions.")
                break

            # Filter input through Llama Guard - returns "SAFE" or "UNSAFE"
            filtered_result = self.guard_filter.filter(user_query)

            if filtered_result in ["safe", "unsafe S6", "unsafe S7"]:  # You need to by pass some cases like "S6" and "S7" so that it can work effectively.
                # Process the user query using the RAG workflow
                try:
                    response = self.nutrition_bot.handle_customer_query(user_id=user_id, query=user_query)  # Call chatbot handler function
                    print(f"Agent: {response}\n")

                except Exception as e:
                    print("Agent: Sorry, I encountered an error while processing your query. Please try again.")
                    print(f"Error: {e}\n")
            else:
                print(
                    "Agent: I apologize, but I cannot process that input as it may be inappropriate. Please try again.")
