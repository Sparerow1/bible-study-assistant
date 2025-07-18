# LangChain Gemini Integration

This project should feature a chatbot that can interact with the Gemini platform, using the LangChain library.

It is specifically trained with vector data from the Bible and other sources, and can provide a simple and engaging interface for users.

## Features

- Simple chat interactions with Gemini
- Conversation memory
- Custom system prompts
- Prompt templates
- Document Q&A with vector embeddings
- Streaming responses

## Using the packaged exe file

1. Simply go to the left hand side, locate the releases section

2. Click on Biblebot-1

3. Click on BibleBot-Windows.zip

4. It will download the BibleBot-Windows.zip

5. Extract the .zip file, and run the .exe file

6. You must use a Windows machine for this to work

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

When initializing the llm object in the config/config_setup_class.py file, you can pass in the following options:

```python
    def __init__(self):
        self.embedding_dimension = 768 # Dimension of the embedding vectors
        self.retriever_k = 15  # number of records to retrieve from the database
        self.llm_temperature = 0.7 # Temperature for response creativity
        self.llm_model = "gemini-2.0-flash-lite" # the llm model to connect to
        self.embedding_model = "models/embedding-001" # the embedding model to use

```
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

## Uploading file to Pinecone Vector database
 
The app itself does not have the capability to upload files to the Pinecone vector database. It is an external functionality that needs to be manually run by the user.

In the upload directory, you will find two scripts:
- `upload_to_pinecone.py` - This script uploads the txt files to the Pinecone vector database.
    - Make sure to specify the txt file and the index name in the script.
    ```python
        INDEX_NAME = "index-name" 
        # The path to the text file to upload.
        SOURCE_FILE_PATH = "content.txt" # specify the path of the text file
    ```
- `pdf_upload.py` - This script uploads PDF files to the Pinecone vector database.
    - This file uploads all the pdf files in the directory you have specified.
    - You must specify the directory path and the index name in the script.
    ```python
    PDF_DIRECTORY = "content/pdf_files"
    PINECONE_INDEX_NAME = "bible-index-pdf" # The number of documents to upload in a single batch.
    ```

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
pip install -r requirements-dev.txt
```

### Rate Limiting

If you hit rate limits, you can:
1. Add delays between requests
2. Implement retry logic
3. Check your API quota in Google AI Studio
4. Wait a while before asking the question later
5. If you have exceeded your token limit, try to lower the `retriever_k` parameter in the config file.
