#!/usr/bin/env python3
"""
Quick test script to verify LangChain setup
Run this to make sure everything is working correctly
"""

import os
from dotenv import load_dotenv

def test_environment():
    """Test that the environment is set up correctly."""
    print("🔧 Testing LangChain Environment Setup...")
    print("-" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check Python version
    import sys
    print(f"✅ Python version: {sys.version}")
    
    # Check if packages are installed
    try:
        import langchain
        print(f"✅ LangChain version: {langchain.__version__}")
    except ImportError:
        print("❌ LangChain not installed")
        return False
    
    try:
        import langchain_openai
        print("✅ LangChain OpenAI integration available")
    except ImportError:
        print("❌ LangChain OpenAI not installed")
        return False
    
    try:
        import streamlit
        print(f"✅ Streamlit available")
    except ImportError:
        print("❌ Streamlit not installed")
        return False
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        if api_key.startswith("sk-") and len(api_key) > 20:
            print("✅ OpenAI API key found and looks valid")
        else:
            print("⚠️ OpenAI API key found but format looks suspicious")
    else:
        print("❌ OpenAI API key not found in .env file")
        print("   Please add OPENAI_API_KEY=your_key_here to .env file")
        return False
    
    print("\n🎉 Environment setup looks good!")
    print("\nNext steps:")
    print("1. Run 'python chat_app.py' for command-line chat")
    print("2. Run 'streamlit run streamlit_app.py' for web interface")
    print("3. Run 'python advanced_chat_app.py' for advanced features")
    
    return True

def test_basic_langchain():
    """Test basic LangChain functionality."""
    print("\n🧪 Testing Basic LangChain Functionality...")
    print("-" * 40)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Test a simple query
        print("Sending test message to OpenAI...")
        response = llm.invoke([HumanMessage(content="Say 'Hello from LangChain!' in a friendly way.")])
        print(f"✅ Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LangChain: {e}")
        return False

if __name__ == "__main__":
    print("🚀 LangChain Chat App - Environment Test")
    print("=" * 50)
    
    # Test environment setup
    env_ok = test_environment()
    
    if env_ok:
        # Test basic functionality
        try:
            test_basic_langchain()
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete!")
