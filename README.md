# LangChain Gemini Integration

This project demonstrates how to use Google's Gemini AI model with LangChain in Python.

## Features

- Simple chat interactions with Gemini
- Conversation memory
- Custom system prompts
- Prompt templates
- Document Q&A with vector embeddings
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
If you are using Chroma, ignore the next step. Follow only if you are using Pinecone for vector database.

```
# Get your Pinecone API key from: https://app.pinecone.io/
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone environment (region) - check your Pinecone dashboard
PINECONE_ENVIRONMENT=us-east-1-aws

# Pinecone index name (will be created automatically)
PINECONE_INDEX_NAME=langchain-gemini
```
## Usage

### Bible Chat

Run the interactive chat script:

```bash
python main.py
```

## Configuration Options

When initializing the llm object, you can pass in the following options:

```python
bot = GeminiLangChainBot(
    api_key="your_key",           # Optional if set in environment
    model_name="gemini-pro-2.5"       # Model to use
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

