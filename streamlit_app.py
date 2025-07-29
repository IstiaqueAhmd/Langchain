#!/usr/bin/env python3
"""
LangChain Chat App - Streamlit Web Interface
A web-based chat interface using Streamlit and LangChain
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="LangChain Chat App",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    if "prompt" not in st.session_state:
        st.session_state.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Be friendly and helpful in your responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    
    if "chain" not in st.session_state:
        st.session_state.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: st.session_state.memory.chat_memory.messages
            )
            | st.session_state.prompt 
            | st.session_state.llm 
            | StrOutputParser()
        )

def main():
    """Main Streamlit app function."""
    st.title("🤖 LangChain Chat App")
    st.write("Chat with an AI assistant powered by LangChain")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Please set your OPENAI_API_KEY in the .env file")
        st.info("You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar with controls
    with st.sidebar:
        st.header("Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chat app demonstrates:
        - **LangChain** for AI orchestration
        - **Memory** for conversation history
        - **Prompt templates** for consistent responses
        - **Streamlit** for web interface
        """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from LangChain
                    response = st.session_state.chain.invoke({"input": prompt})
                    
                    # Update memory
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(response)
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
