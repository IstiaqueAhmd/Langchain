# LangChain Chat App

A collection of simple chat applications built with LangChain, demonstrating various features and capabilities of the framework.

## 🚀 Features

### Basic Chat App (`chat_app.py`)
- Simple command-line chat interface
- Conversation memory
- OpenAI integration
- Basic error handling

### Streamlit Web App (`streamlit_app.py`)
- Beautiful web interface
- Real-time chat
- Conversation history
- Clear chat functionality

### Advanced Chat App (`advanced_chat_app.py`)
- Tool integration (calculator, time, weather)
- Enhanced memory with conversation summaries
- Function calling capabilities
- Multiple model support

## 📋 Prerequisites

1. Python 3.8 or higher
2. OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/api-keys))

## 🛠️ Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Langchain
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   - Copy the `.env` file
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## 🎯 Usage

### Run the Basic Chat App
```bash
python chat_app.py
```

### Run the Streamlit Web App
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

### Run the Advanced Chat App
```bash
python advanced_chat_app.py
```

## 💡 Key LangChain Concepts Demonstrated

### 1. **Chat Models**
Using OpenAI's chat models through LangChain's unified interface:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
```

### 2. **Memory**
Storing conversation history for context:
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)
```

### 3. **Prompt Templates**
Structured prompts with variables and conversation history:
```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
```

### 4. **Chains**
Connecting components together:
```python
chain = prompt | llm | StrOutputParser()
```

### 5. **Tools and Agents**
Giving the AI access to external functions:
```python
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent

tools = [Tool(name="calculator", func=calculate, description="Calculate math")]
agent = create_openai_functions_agent(llm, tools, prompt)
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ANTHROPIC_API_KEY`: For Claude models (optional)
- `GOOGLE_API_KEY`: For Google models (optional)

### Model Selection
You can easily switch between different models:
- `gpt-3.5-turbo`: Fast and cost-effective
- `gpt-4`: More capable but slower and more expensive
- `gpt-4-turbo`: Latest GPT-4 model

## 📁 Project Structure

```
Langchain/
├── .env                    # Environment variables
├── chat_app.py            # Basic command-line chat app
├── streamlit_app.py       # Web-based chat interface
├── advanced_chat_app.py   # Advanced app with tools
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 Customization

### Adding New Tools
In `advanced_chat_app.py`, add new tools to the `_create_tools()` method:

```python
def my_custom_tool(input_text: str) -> str:
    # Your custom logic here
    return f"Processed: {input_text}"

tools.append(Tool(
    name="my_tool",
    func=my_custom_tool,
    description="Description of what this tool does"
))
```

### Changing the System Prompt
Modify the system message in the prompt templates to change the AI's behavior:

```python
("system", "You are a helpful assistant specialized in [your domain].")
```

### Adding Different LLM Providers
You can easily add support for other providers:

```python
# For Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# For Google models
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro")
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains a valid OpenAI API key
2. **Import Error**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Memory Issues**: For long conversations, the advanced app uses summary memory to handle token limits

### Getting Help

- Check the [LangChain Documentation](https://python.langchain.com/)
- Review [OpenAI API Documentation](https://platform.openai.com/docs)
- Create an issue in this repository

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔮 Next Steps

Ideas for extending this chat app:
- Add document upload and Q&A functionality
- Implement RAG (Retrieval-Augmented Generation)
- Add voice input/output
- Create a mobile app interface
- Add user authentication
- Implement conversation sharing
- Add support for image analysis
