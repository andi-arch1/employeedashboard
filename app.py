import streamlit as st
import requests

# Config
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3"  # or your preferred model

st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖")
st.title("🧠 Simple Q&A with Ollama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send to Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": st.session_state.messages,
            "stream": False,
        }
    )

    # Extract and show response
    if response.status_code == 200:
        bot_message = response.json()["message"]["content"]
        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
    else:
        st.error("❌ Failed to get response from Ollama.")
