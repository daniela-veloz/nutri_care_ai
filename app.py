import streamlit as st
from dotenv import load_dotenv
import os
import logging

from src.agentic.agent.nutri_agent import NutritionAgent
from src.rate_limit.rate_limiter import RateLimiter, RateLimitType


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nutri_care_ai.log'),
            logging.StreamHandler()
        ]
    )


def load_and_validate_environment() -> bool:
    """
    Load environment variables and validate that all required API keys are present.

    Returns:
        bool: True if all required API keys are present and valid

    Raises:
        ValueError: If any required API key is missing or empty
    """
    try:
        load_dotenv(override=True)
        
        required_keys = {
            'OPENAI_API_KEY': 'OpenAI API',
            'GROQ_API_KEY': 'Groq API',
            'MEM0_API_KEY': 'Mem0 API',
        }

        missing_keys = [
            f"{key} ({service})"
            for key, service in required_keys.items()
            if not os.getenv(key, '').strip()
        ]

        if missing_keys:
            error_msg = f"Missing required API keys: {', '.join(missing_keys)}"
            logging.error(error_msg)
            st.error(error_msg)
            st.stop()
            
        logging.info("Environment variables loaded successfully")
        return True
        
    except Exception as e:
        error_msg = f"Failed to load environment: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        st.stop()

@st.cache_resource
def create_agent() -> NutritionAgent:
    """Create and cache the nutrition agent instance."""
    try:
        logging.info("Initializing nutrition agent")
        agent = NutritionAgent()
        logging.info("Nutrition agent initialized successfully")
        return agent
    except Exception as e:
        error_msg = f"Failed to initialize nutrition agent: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        st.stop()


@st.cache_resource
def create_rate_limiter() -> RateLimiter:
    """Create and cache the rate limiter instance."""
    try:
        logging.info("Initializing rate limiter")
        rate_limiter = RateLimiter()
        logging.info("Rate limiter initialized successfully")
        return rate_limiter
    except Exception as e:
        error_msg = f"Failed to initialize rate limiter: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        st.stop()

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = None


def handle_login() -> bool:
    """Handle user login form and return True if user is logged in."""
    if st.session_state.user_id is not None:
        return True
        
    with st.form("login_form", clear_on_submit=True):
        user_id = st.text_input("Please enter your name to begin:")
        submit_button = st.form_submit_button("Login")
        
        if submit_button and user_id.strip():
            st.session_state.user_id = user_id.strip()
            logging.info(f"User logged in: {user_id}")
            welcome_msg = f"Welcome, {user_id}! How can I help you with nutrition disorders today?"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": welcome_msg
            })
            st.rerun()
            
    return False


def display_chat_history() -> None:
    """Display the chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def check_rate_limit() -> bool:
    """Check if user is within rate limits and display appropriate messages."""
    # Initialize rate limiter if not already done
    if st.session_state.rate_limiter is None:
        st.session_state.rate_limiter = create_rate_limiter()
    
    # Check rate limit
    rate_result = st.session_state.rate_limiter.check_rate_limit()
    
    if not rate_result.valid:
        if rate_result.limit_type == RateLimitType.HOURLY_LIMIT:
            st.error(f"⏱️ Hourly limit reached! Please wait {rate_result.next_reset} minutes before your next request.")
        elif rate_result.limit_type == RateLimitType.DAILY_LIMIT:
            st.error(f"📅 Daily limit reached! Please wait {rate_result.next_reset} hours before your next request.")
        
        # Display usage stats
        with st.expander("Usage Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hourly Usage", 
                         f"{rate_result.stats['hourly_used']}/{rate_result.stats['hourly_limit']}")
            with col2:
                st.metric("Daily Usage", 
                         f"{rate_result.stats['daily_used']}/{rate_result.stats['daily_limit']}")
        
        logging.warning(f"Rate limit exceeded for user {st.session_state.user_id}: {rate_result.limit_type}")
        return False
    
    return True


def display_project_info() -> None:
    """Display comprehensive project information in sidebar."""
    with st.sidebar:
        st.markdown("# 🥗 NutriCare AI")
        st.markdown("### Intelligent Nutrition Disorder Specialist")
        
        # Project Purpose
        with st.expander("📋 About This Application", expanded=True):
            st.markdown("""
            **NutriCare AI** is an intelligent nutrition disorder specialist that provides:
            
            - **Personalized nutrition guidance** for various health conditions
            - **Evidence-based recommendations** using RAG technology  
            - **Memory-enhanced conversations** that remember your preferences
            - **Professional medical support** with empathetic responses
            
            This AI assistant specializes in nutrition disorders and can help with 
            dietary planning, nutritional deficiencies, and health-conscious meal recommendations.
            """)
        
        # How to Use
        with st.expander("🚀 How to Use"):
            st.markdown("""
            **Getting Started:**
            1. Enter your name to begin a session
            2. Ask questions about nutrition and health
            3. The AI remembers your conversation history
            4. Type 'exit' to end your session
            
            **Example Questions:**
            - "What should I eat for iron deficiency?"
            - "Help me plan meals for diabetes management"
            - "What foods should I avoid with high cholesterol?"
            - "Suggest a balanced diet for weight management"
            - "What are the best sources of protein for vegetarians?"
            """)
        
        # Rate Limits Information
        with st.expander("⏱️ Usage Limits"):
            st.markdown("""
            **Rate Limits (per IP address):**
            - **Hourly Limit:** 10 requests
            - **Daily Limit:** 25 requests
            
            **Why Rate Limits?**
            - Ensures fair access for all users
            - Prevents system overload
            - Maintains response quality
            
            Your usage resets automatically each hour/day.
            """)
        
        # Features
        with st.expander("✨ Key Features"):
            st.markdown("""
            **AI Capabilities:**
            - 🧠 **Memory System** - Remembers your preferences
            - 📚 **Knowledge Base** - Evidence-based nutrition information
            - 🔍 **Smart Retrieval** - Finds relevant information quickly
            - 🛡️ **Content Filtering** - Ensures safe and appropriate responses
            
            **Technical Stack:**
            - LangChain for AI orchestration
            - OpenAI GPT models for responses
            - ChromaDB for knowledge storage
            - Mem0 for conversation memory
            """)
        
        st.divider()


def display_usage_stats() -> None:
    """Display current usage statistics in sidebar."""
    if st.session_state.rate_limiter is None:
        return
        
    rate_result = st.session_state.rate_limiter.check_rate_limit()
    
    with st.sidebar:
        st.markdown("### 📊 Your Usage Statistics")
        
        # Hourly usage
        hourly_percentage = (rate_result.stats['hourly_used'] / rate_result.stats['hourly_limit']) * 100
        st.metric(
            "Hourly Requests", 
            f"{rate_result.stats['hourly_used']}/{rate_result.stats['hourly_limit']}",
            delta=f"{rate_result.stats['hourly_remaining']} remaining"
        )
        st.progress(hourly_percentage / 100)
        
        # Daily usage
        daily_percentage = (rate_result.stats['daily_used'] / rate_result.stats['daily_limit']) * 100
        st.metric(
            "Daily Requests", 
            f"{rate_result.stats['daily_used']}/{rate_result.stats['daily_limit']}",
            delta=f"{rate_result.stats['daily_remaining']} remaining"
        )
        st.progress(daily_percentage / 100)
        
        # Usage tips
        if hourly_percentage > 80 or daily_percentage > 80:
            st.warning("⚠️ Approaching usage limit! Plan your questions wisely.")
        elif hourly_percentage > 50 or daily_percentage > 50:
            st.info("ℹ️ You're halfway through your limit.")
        
        st.divider()
        
        # Contact and support
        st.markdown("### 🆘 Need Help?")
        st.markdown("""
        **Tips for Better Results:**
        - Be specific about your health conditions
        - Mention dietary restrictions or preferences
        - Ask follow-up questions for clarification
        
        **Remember:** This AI provides general guidance only. 
        Always consult healthcare professionals for medical advice.
        """)


def handle_user_input() -> None:
    """Handle user input and generate responses."""
    user_query = st.chat_input("Ask me anything about nutrition disorders (or type 'exit' to end)")
    
    if not user_query:
        return
        
    if user_query.lower() == "exit":
        handle_exit(user_query)
        return
    
    # Check rate limits before processing
    if not check_rate_limit():
        return
        
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.write(user_query)
    
    # Initialize agent if not already done
    if st.session_state.agent is None:
        st.session_state.agent = create_agent()
    
    # Generate response
    try:
        logging.info(f"Processing query from user {st.session_state.user_id}: {user_query}")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.handle_customer_query(
                    st.session_state.user_id, user_query
                )
            st.write(response)
            
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Record successful request for rate limiting
        from src.rate_limit.ip_extractor import IPExtractor
        ip_address = IPExtractor.get_client_ip()
        st.session_state.rate_limiter.record_request(ip_address)
        logging.info(f"Response generated successfully for user {st.session_state.user_id}")
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        logging.error(f"Error processing query for user {st.session_state.user_id}: {str(e)}")
        st.error(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


def handle_exit(user_query: str) -> None:
    """Handle user exit command."""
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.write(user_query)
        
    goodbye_msg = "Goodbye! Feel free to return if you have more questions about nutrition disorders."
    st.session_state.chat_history.append({"role": "assistant", "content": goodbye_msg})
    
    with st.chat_message("assistant"):
        st.write(goodbye_msg)
        
    # Reset session
    logging.info(f"User {st.session_state.user_id} logged out")
    st.session_state.user_id = None
    st.session_state.agent = None
    st.rerun()


def main() -> None:
    """Main Streamlit application."""
    # Setup logging first
    setup_logging()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="NutriCare AI - Nutrition Disorder Specialist",
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    logging.info("Streamlit application started")
    
    # Load and validate environment
    load_and_validate_environment()
    
    # Initialize session state
    initialize_session_state()
    
    # Display project information in sidebar (always visible)
    display_project_info()
    
    # Handle login
    if not handle_login():
        return
        
    # Display chat interface for logged-in users
    display_usage_stats()
    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    main()

