import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

def escape_curly_braces(text: str) -> str:
    # Escape curly braces for any literal usage
    return text.replace("{", "{{").replace("}", "}}")

# CSS styling with Open Sans and improved light/dark themes (Apple-inspired)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
    body {
        font-family: 'Open Sans', sans-serif;
        margin: 0;
        padding: 0;
        transition: background-color 0.3s, color 0.3s;
    }
    .main-container {
        padding: 2rem;
        transition: background-color 0.3s, color 0.3s;
    }
    .sidebar-container {
        padding: 2rem 1rem;
        transition: background-color 0.3s, color 0.3s;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .chat-container {
        max-width: 800px;
        margin: auto;
    }
    .stTextInput textarea {
        transition: background-color 0.3s, color 0.3s;
    }
    /* Light theme styling */
    .light-theme body,
    .light-theme .main-container {
        background-color: #FFFFFF;
        color: #1C1C1E;
    }
    .light-theme .sidebar-container {
        background-color: #F2F2F7;
        color: #1C1C1E;
    }
    .light-theme .stTextInput textarea {
        background-color: #FFFFFF !important;
        color: #1C1C1E !important;
    }
    /* Dark theme styling */
    .dark-theme body,
    .dark-theme .main-container {
        background-color: #1C1C1E;
        color: #FFFFFF;
    }
    .dark-theme .sidebar-container {
        background-color: #2C2C2E;
        color: #FFFFFF;
    }
    .dark-theme .stTextInput textarea {
        background-color: #2C2C2E !important;
        color: #FFFFFF !important;
    }
    /* Chat message styling */
    .stChatMessage {
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .light-theme .stChatMessage.user {
        background-color: #E5E5EA;
        color: #1C1C1E;
    }
    .light-theme .stChatMessage.ai {
        background-color: #D1D1D6;
        color: #1C1C1E;
    }
    .dark-theme .stChatMessage.user {
        background-color: #3A3A3C;
        color: #FFFFFF;
    }
    .dark-theme .stChatMessage.ai {
        background-color: #48484A;
        color: #FFFFFF;
    }
    /* Copyright block styling */
    .sidebar-copyright {
        font-size: 0.8rem;
        margin-top: 2rem;
        text-align: left;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    theme_choice = st.selectbox("Theme", ["Light", "Dark"], index=0)
    theme_class = "light-theme" if theme_choice == "Light" else "dark-theme"
    st.markdown(f'<div class="sidebar-container {theme_class}">', unsafe_allow_html=True)
    st.markdown("<div class='sidebar-header'>⚙️ Configuration</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b"], index=0)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert  
    - 🐞 Debugging Assistant  
    - 📝 Code Documentation  
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    st.markdown("<div class='sidebar-copyright'>© 2023 Shreyas Prabhakar</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Wrap the main container with the same theme class
st.markdown(f'<div class="main-container {theme_class}">', unsafe_allow_html=True)

# App Title and Caption in main container
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Initialize the chat engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management for storing chat history
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm Deepseek. How can I help you code today? 💻"}]

# Chat container: display conversation
with st.container():
    for message in st.session_state.message_log:
        css_class = "user" if message["role"] == "user" else "ai"
        with st.chat_message(message["role"], avatar=None):
            st.markdown(f'<div class="stChatMessage {css_class}">{message["content"]}</div>', unsafe_allow_html=True)

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        # Escape any curly braces in message content so they are not treated as variables.
        content = escape_curly_braces(msg["content"])
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(content))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(content))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()

# End main container
st.markdown("</div>", unsafe_allow_html=True)