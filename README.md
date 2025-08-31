# BibleBot - AI-Powered Biblical Q&A System

A modern chatbot system that provides intelligent answers to biblical questions using LangChain, Google Gemini, and vector embeddings. Features a FastAPI backend with both HTML and PHP frontend options.

## Features

- **AI-Powered Q&A**: Intelligent responses using Google Gemini 2.0 Flash Lite
- **Vector Search**: Semantic search through biblical content using embeddings
- **Modern Web Interface**: Beautiful, responsive chat UI with real-time communication
- **Multiple Frontends**: Choose between HTML (built-in) or PHP frontend
- **Conversation Memory**: Maintains context across chat sessions
- **Source References**: Displays biblical references and sources for answers
- **Health Monitoring**: Real-time status indicators and health checks
- **Statistics Display**: Shows vector count and dimension information
- **Docker Support**: Easy deployment with Docker containerization

## Architecture

- **Backend**: FastAPI (Python) with LangChain and Gemini integration
- **Vector Database**: Pinecone or Chroma for semantic search
- **Frontend Options**:
  - Built-in HTML interface (served by FastAPI)
  - Separate PHP frontend served by FastAPI (optional)
- **Deployment**: Docker container with both Python and PHP support

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd biblebot
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Build and run with Docker**:
   ```bash
   docker build -t biblebot .
   docker run -p 8000:8000 --env-file .env biblebot
   ```

4. **Access the application**:
   - **FastAPI Backend**: http://localhost:8000
   - **HTML Frontend**: http://localhost:8000/web
   - **API Documentation**: http://localhost:8000/docs
   - **PHP Frontend**: http://localhost:8080 (see PHP setup below)

### Manual Setup

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

#### 3. Configure Environment

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_google_api_key_here` with your actual API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

If using Pinecone for vector database (optional):
```
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=langchain-gemini
```

#### 4. Start the FastAPI Backend

```bash
python start_web_service.py
```

The FastAPI server will be available at:
- **Web Interface**: http://localhost:8000/web
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

#### 5. Optional: Setup PHP Frontend

For a separate PHP frontend:

```bash
cd php_frontend
php -S localhost:8080
```

Then access the PHP frontend at http://localhost:8080

## Usage

### Command Line Interface

Run the interactive chat script:

```bash
python main.py
```

### Web Interface

1. **HTML Frontend** (built-in): Access http://localhost:8000/web
2. **PHP Frontend** (optional): Access http://localhost:8080

Both frontends provide:
- Real-time chat interface
- Health status monitoring
- Vector database statistics
- Memory management
- Source reference display

### API Endpoints

The FastAPI backend provides these endpoints:

- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /stats` - Vector database statistics
- `POST /chat` - Send chat messages
- `POST /clear-memory` - Clear conversation memory
- `GET /docs` - Interactive API documentation

## Configuration

### Backend Configuration

Edit `config/config_setup_class.py` to customize:

```python
def __init__(self):
    self.embedding_dimension = 768  # Dimension of embedding vectors
    self.retriever_k = 15          # Number of records to retrieve
    self.llm_temperature = 0.7     # Response creativity
    self.llm_model = "gemini-2.0-flash-lite"  # LLM model
    self.embedding_model = "models/embedding-001"  # Embedding model
```

### PHP Frontend Configuration

Edit `php_frontend/config.php` to customize:

```php
define('API_BASE_URL', 'http://localhost:8000');  // FastAPI server URL
define('CHAT_TITLE', 'BibleBot (PHP)');
define('API_TIMEOUT', 30);
```

## Vector Database Setup

### Uploading Content

The app includes scripts to upload content to your vector database:

#### Text Files
```bash
cd upload
python upload_to_pinecone.py
```
Edit the script to specify your file and index name.

#### PDF Files
```bash
cd upload
python pdf_upload.py
```
Edit the script to specify your directory and index name.

### Database Options

- **Chroma** (default): Local vector database, no additional setup required
- **Pinecone**: Cloud vector database, requires API key and environment setup

## Deployment

### Production Deployment

1. **Environment Variables**: Set production environment variables
2. **CORS Configuration**: Update CORS settings in `api/main.py`
3. **Security**: Disable debug mode and enable rate limiting
4. **Web Server**: Use a production web server (Nginx, Apache) for PHP frontend

### Docker Deployment

```bash
# Build production image
docker build -t biblebot:latest .

# Run with environment file
docker run -d -p 8000:8000 --env-file .env --name biblebot biblebot:latest
```

## Troubleshooting

### Common Issues

1. **"Google API key not found"**: Ensure `.env` file exists with correct API key
2. **Import Errors**: Install all dependencies with `pip install -r requirements.txt`
3. **Rate Limiting**: Check API quota in Google AI Studio or reduce `retriever_k` parameter
4. **PHP Frontend Issues**: Ensure cURL extension is enabled and API_BASE_URL is correct

### Debug Mode

Enable debug mode by setting environment variable:
```bash
export DEBUG=true
```

### Logs

- **FastAPI Logs**: Check console output when running `start_web_service.py`
- **PHP Logs**: Check web server error logs or enable logging in `config.php`

## Requirements

- Python 3.8+
- PHP 7.4+ (for PHP frontend)
- Google API key
- Internet connection
- Optional: Pinecone API key for cloud vector database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation at http://localhost:8000/docs
3. Check the PHP frontend README in `php_frontend/README.md`
4. Create an issue in the repository
