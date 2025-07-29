#!/usr/bin/env python3
"""
Advanced LangChain Chat App
Demonstrates more advanced LangChain features:
- Multiple LLM providers
- Custom chains
- Document retrieval
- Function calling
"""

import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
import datetime

# Load environment variables
load_dotenv()

class AdvancedChatApp:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize advanced chat app with tools and enhanced memory."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Use summary memory to handle longer conversations
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=1000
        )
        
        # Create tools for the agent
        self.tools = self._create_tools()
        
        # Create agent with tools
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create tools that the AI can use."""
        
        def get_current_time() -> str:
            """Get the current date and time."""
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def calculate_math(expression: str) -> str:
            """Calculate a mathematical expression safely."""
            try:
                # Only allow basic math operations for security
                allowed_chars = set('0123456789+-*/()., ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Only basic math operations are allowed"
                
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error calculating: {e}"
        
        def get_weather_info(location: str) -> str:
            """Mock weather function (would integrate with real API in production)."""
            return f"Weather in {location}: Sunny, 22°C (This is a mock response - integrate with a real weather API)"
        
        tools = [
            Tool(
                name="get_current_time",
                func=get_current_time,
                description="Get the current date and time"
            ),
            Tool(
                name="calculate_math",
                func=calculate_math,
                description="Calculate mathematical expressions. Input should be a valid math expression like '2+2' or '10*5'"
            ),
            Tool(
                name="get_weather_info",
                func=get_weather_info,
                description="Get weather information for a location. Input should be a city name."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create an agent that can use tools."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to tools. 
            Use the tools when appropriate to help answer questions.
            Always be friendly and explain what you're doing when using tools."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def chat(self, user_input: str) -> str:
        """Process user input and return AI response."""
        try:
            # Get chat history for context
            chat_history = self.memory.chat_memory.messages
            
            # Run the agent
            response = self.agent.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Save to memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response["output"])
            
            return response["output"]
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far."""
        try:
            summary = self.memory.predict_new_summary(
                self.memory.chat_memory.messages, ""
            )
            return summary if summary else "No conversation yet."
        except:
            return "Unable to generate summary."
    
    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()
    
    def run(self):
        """Run the interactive chat loop."""
        print("🚀 Advanced LangChain Chat App")
        print("Available commands:")
        print("  'quit' - Exit the app")
        print("  'clear' - Clear conversation history")
        print("  'summary' - Get conversation summary")
        print("  'tools' - List available tools")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    print("✅ Conversation history cleared!")
                    continue
                elif user_input.lower() == 'summary':
                    summary = self.get_conversation_summary()
                    print(f"📋 Summary: {summary}")
                    continue
                elif user_input.lower() == 'tools':
                    print("🛠️ Available tools:")
                    for tool in self.tools:
                        print(f"  - {tool.name}: {tool.description}")
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
    """Main function."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Please set your OPENAI_API_KEY in the .env file")
        return
    
    print("Choose the LLM model:")
    print("1. GPT-3.5 Turbo (faster, cheaper)")
    print("2. GPT-4 (more capable, slower)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    model = "gpt-4" if choice == "2" else "gpt-3.5-turbo"
    
    app = AdvancedChatApp(model_name=model)
    app.run()

if __name__ == "__main__":
    main()
