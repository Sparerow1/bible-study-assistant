# BibleBot Web Service

This document explains how to run your BibleBot as a FastAPI web service, providing both a REST API and a web interface.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Make sure your `.env` file is properly configured with your API keys:

```bash
# Google API Key for Gemini
GOOGLE_API_KEY=your_google_api_key_here

# Pinecone Configuration (if using Pinecone)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=langchain-gemini
```

### 3. Start the Web Service

#### Option A: Using the startup script (Recommended)
```bash
python start_web_service.py
```

#### Option B: Direct uvicorn command
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Access the Service

- **Web Interface**: http://localhost:8000/web
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🌐 Web Interface

The web interface provides a beautiful, responsive chat interface where users can:

- Ask questions about the Bible
- View real-time responses
- See source documents and references
- Monitor system health and statistics

### Features:
- Modern, responsive design
- Real-time chat experience
- Source document display
- System status indicators
- Mobile-friendly interface

## 🔌 API Endpoints

### Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
    "message": "What does the Bible say about love?",
    "session_id": "optional_session_id"
}
```

**Response:**
```json
{
    "answer": "The Bible teaches that love is the greatest commandment...",
    "session_id": "optional_session_id",
    "sources": [
        {
            "content": "1 Corinthians 13:4-7 - Love is patient, love is kind...",
            "metadata": {"source": "1 Corinthians 13"}
        }
    ]
}
```

### Health Check
```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "message": "BibleBot API is running",
    "vector_count": 15000
}
```

### Statistics
```http
GET /stats
```

**Response:**
```json
{
    "total_vectors": 15000,
    "dimension": 768,
    "index_name": "langchain-gemini"
}
```

### Clear Memory
```http
POST /clear-memory
```

**Response:**
```json
{
    "message": "Memory cleared successfully"
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Port to run the server on |
| `HOST` | 0.0.0.0 | Host to bind the server to |
| `RELOAD` | false | Enable auto-reload for development |
| `DEBUG` | false | Enable debug mode |

### Example Configuration

```bash
# Development with auto-reload
PORT=8000 HOST=localhost RELOAD=true python start_web_service.py

# Production
PORT=8080 HOST=0.0.0.0 python start_web_service.py
```

## 🔧 Development

### Running in Development Mode

```bash
# Enable auto-reload for development
RELOAD=true python start_web_service.py
```

### API Testing

You can test the API using curl:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is faith?"}'

# Get statistics
curl http://localhost:8000/stats
```

### Using the Interactive API Documentation

1. Start the web service
2. Navigate to http://localhost:8000/docs
3. Use the interactive Swagger UI to test all endpoints

## 🚀 Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "start_web_service.py"]
```

Build and run:
```bash
docker build -t biblebot-web .
docker run -p 8000:8000 --env-file .env biblebot-web
```

### Production Considerations

1. **Security**: Configure CORS appropriately for production
2. **Environment**: Use proper environment variables
3. **Logging**: Implement proper logging
4. **Monitoring**: Add health checks and monitoring
5. **Rate Limiting**: Consider implementing rate limiting
6. **SSL**: Use HTTPS in production

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change the port
   PORT=8001 python start_web_service.py
   ```

2. **API key not found**
   - Ensure your `.env` file is properly configured
   - Check that the API key is valid

3. **Pinecone connection issues**
   - Verify your Pinecone API key and environment
   - Check that your index exists and has data

4. **Memory issues**
   - Clear conversation memory using the `/clear-memory` endpoint
   - Restart the service if needed

### Debug Mode

Enable debug mode for more detailed error information:

```bash
DEBUG=true python start_web_service.py
```

## 📊 Monitoring

The web service provides several monitoring endpoints:

- `/health` - Overall service health
- `/stats` - Vector database statistics
- `/docs` - API documentation with testing interface

## 🔄 Migration from CLI

If you're migrating from the command-line interface:

1. **Same Core Logic**: The web service uses the same `BibleQASystem` class
2. **Same Configuration**: Uses the same `.env` file and configuration
3. **Same Vector Database**: Connects to the same Pinecone index
4. **Enhanced Features**: Adds web interface and API endpoints

## 📝 API Examples

### Python Client Example

```python
import requests

# Initialize client
base_url = "http://localhost:8000"

# Ask a question
response = requests.post(f"{base_url}/chat", json={
    "message": "What does the Bible say about forgiveness?",
    "session_id": "my_session"
})

if response.status_code == 200:
    result = response.json()
    print(f"Answer: {result['answer']}")
    if result.get('sources'):
        print(f"Sources: {len(result['sources'])} references found")
else:
    print(f"Error: {response.status_code}")
```

### JavaScript Client Example

```javascript
// Ask a question
async function askBibleBot(question) {
    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: question,
                session_id: 'web_session'
            })
        });
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}

// Usage
askBibleBot("What is the meaning of grace?").then(result => {
    console.log(result.answer);
});
```

## 🎯 Next Steps

1. **Authentication**: Add user authentication and session management
2. **Rate Limiting**: Implement request rate limiting
3. **Caching**: Add response caching for common questions
4. **Analytics**: Track usage patterns and popular questions
5. **Multi-language**: Support for multiple languages
6. **Mobile App**: Create a mobile application using the API

---

For more information about the core BibleBot functionality, see the main [README.md](README.md).
