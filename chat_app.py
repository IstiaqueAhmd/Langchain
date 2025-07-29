#!/usr/bin/env python3
"""
Simple LangChain Chat App - Command Line Version
This demonstrates basic LangChain concepts including:
- Chat models
- Prompt templates
- Memory for conversation history
- Chain of operations
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

class SimpleChatApp:
    def __init__(self):
        """Initialize the chat app with LangChain components."""
        # Initialize the LLM (Large Language Model)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize memory to store conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create a prompt template with system message and conversation history
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Be friendly and helpful in your responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create the conversation chain
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.chat_memory.messages
            )
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def chat(self, user_input: str) -> str:
        """Process user input and return AI response."""
        try:
            # Get response from the chain
            response = self.chain.invoke({"input": user_input})
            
            # Save the conversation to memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()
        print("Conversation history cleared!")
    
    def run(self):
        """Run the interactive chat loop."""
        print("🤖 LangChain Chat App")
        print("Type 'quit' to exit, 'clear' to clear history")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif not user_input:
                    continue
                
                print("🤖 AI: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function to run the chat app."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Please set your OPENAI_API_KEY in the .env file")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Create and run the chat app
    app = SimpleChatApp()
    app.run()

if __name__ == "__main__":
    main()
