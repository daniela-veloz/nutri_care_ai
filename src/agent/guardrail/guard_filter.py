# Import libraries for handling user input filtering and accessing user data
from groq import Groq  # Llama Guard client for filtering user input
from google.colab import userdata  # Access user-specific data securely

# Retrieve the Llama API key from user data
groq_api_key = userdata.get('LLAMA_GUARD_API')

# Initialize the Llama Guard client with the API key
llama_guard_client = Groq(api_key=groq_api_key)

# Function to filter user input with Llama Guard
def filter_input_with_llama_guard(user_input, model="llama-guard-3-8b"):
    """
    Filters user input using Llama Guard to ensure it is safe.

    Parameters:
    - user_input: The input provided by the user.
    - model: The Llama Guard model to be used for filtering (default is "llama-guard-3-8b").

    Returns:
    - The filtered and safe input.
    """
    try:
        # Create a request to Llama Guard to filter the user input
        response = llama_guard_client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model=model,
        )
        # Return the filtered input
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Llama Guard: {e}")
        return None