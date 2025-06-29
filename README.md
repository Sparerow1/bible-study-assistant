# LangChain Gemini Integration

This project demonstrates how to use Google's Gemini AI model with LangChain in Python.

## Features

- Simple chat interactions with Gemini
- Conversation memory
- Custom system prompts
- Prompt templates
- Document Q&A with vector embeddings
- Asynchronous chat support
- Streaming responses

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Configure Environment

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_google_api_key_here` with your actual API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

## Usage

### Simple Chat

Run the interactive chat script:

```bash
python simple_chat.py
```

### Advanced Usage

The main `langchain_gemini.py` file contains a comprehensive `GeminiLangChainBot` class with the following features:

```python
from langchain_gemini import GeminiLangChainBot

# Initialize the bot
bot = GeminiLangChainBot()

# Simple chat
response = bot.simple_chat("Hello, how are you?")

# Chat with system prompt
response = bot.chat_with_system_prompt(
    "Explain quantum physics", 
    "You are a physics professor"
)

# Conversation with memory
bot.conversation_chat("My name is Alice")
response = bot.conversation_chat("What's my name?")

# Document Q&A (requires text files in a directory)
qa_chain = bot.setup_document_qa("./documents")
result = qa_chain({"query": "What is the main topic?"})
```

### Custom Prompt Templates

```python
template = """
You are an expert {field} tutor. 
Explain {concept} to a beginner.

Concept: {concept}
Field: {field}

Explanation:
"""

chain = bot.create_prompt_template_chain(template, ["field", "concept"])
response = chain.run(field="programming", concept="variables")
```

## Available Methods

### GeminiLangChainBot Class

- `simple_chat(message)` - Basic chat without memory
- `chat_with_system_prompt(message, system_prompt)` - Chat with custom system prompt
- `conversation_chat(message)` - Chat with conversation memory
- `create_prompt_template_chain(template, variables)` - Create custom prompt chains
- `setup_document_qa(documents_path)` - Set up document Q&A system
- `async_chat(message)` - Asynchronous chat
- `clear_memory()` - Clear conversation memory
- `get_conversation_history()` - Get chat history

## Models Available

- `gemini-pro` - Text generation (default)
- `gemini-pro-vision` - Text and image understanding
- `models/embedding-001` - Text embeddings

## Configuration Options

When initializing `GeminiLangChainBot`, you can customize:

```python
bot = GeminiLangChainBot(
    api_key="your_key",           # Optional if set in environment
    model_name="gemini-pro"       # Model to use
)
```

The LLM can be configured with:
- `temperature` - Response creativity (0.0 to 1.0)
- `streaming` - Enable streaming responses
- `convert_system_message_to_human` - Convert system messages for Gemini

## Error Handling

The bot includes error handling for common issues:
- Missing API key
- Network errors
- Invalid model names
- Rate limiting

## Requirements

- Python 3.8+
- Google API key
- Internet connection

## Troubleshooting

### "Google API key not found" Error

Make sure you have:
1. Created a `.env` file with your API key
2. Set the `GOOGLE_API_KEY` environment variable
3. Or passed the API key directly to the constructor

### Import Errors

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Rate Limiting

If you hit rate limits, you can:
1. Add delays between requests
2. Implement retry logic
3. Check your API quota in Google AI Studio

## License

This project is for educational purposes. Check Google's terms of service for commercial use.
# biblebot
