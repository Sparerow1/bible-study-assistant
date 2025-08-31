import os
import sys
import traceback
import subprocess
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa.qa_class import BibleQASystem
from config.config_setup_class import EnvironmentValidator
from pathHandling.validation import Validation

# Load environment variables
env_path = Validation.load_env_from_bundle()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="BibleBot API",
    description="A web service for Bible Q&A using LangChain and Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Global Bible Q&A system instance
bible_qa_system: Optional[BibleQASystem] = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: Optional[List[dict]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    vector_count: Optional[int] = None

class StatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    index_name: str

@app.on_event("startup")
async def startup_event():
    """Initialize the Bible Q&A system on startup."""
    global bible_qa_system
    
    try:
        print("🚀 Starting BibleBot API...")
        
        # Validate environment
        if not EnvironmentValidator.validate_environment():
            raise Exception("Environment validation failed")
        
        # Initialize Bible Q&A system
        bible_qa_system = BibleQASystem()
        if not bible_qa_system.initialize():
            raise Exception("Failed to initialize Bible Q&A system")
        
        print("✅ BibleBot API initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize BibleBot API: {e}")
        if os.getenv("DEBUG", "").lower() == "true":
            traceback.print_exc()
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    global bible_qa_system
    
    if bible_qa_system is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        vector_count = bible_qa_system.pinecone_manager.get_vector_count()
        return HealthResponse(
            status="healthy",
            message="BibleBot API is running",
            vector_count=vector_count
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            message=f"Service error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()

@app.get("/web")
async def web_interface():
    """Serve the PHP web interface."""
    return await serve_php_file("php_frontend/index.php")

@app.post("/web")
async def web_interface_post(request: Request):
    """Handle PHP form submissions directly."""
    form_data = await request.form()
    action = form_data.get('action')
    
    if action == 'chat':
        return await handle_php_chat(form_data)
    elif action == 'health':
        return await handle_php_health()
    elif action == 'stats':
        return await handle_php_stats()
    elif action == 'clear_memory':
        return await handle_php_clear_memory()
    else:
        # Fall back to PHP execution for other cases
        return await serve_php_file("php_frontend/index.php")

@app.get("/web/{path:path}")
async def web_interface_path(path: str):
    """Serve PHP files from the web interface."""
    php_file_path = f"php_frontend/{path}"
    return await serve_php_file(php_file_path)

@app.post("/web/{path:path}")
async def web_interface_path_post(path: str, request: Request):
    """Handle PHP form submissions for specific paths."""
    form_data = await request.form()
    action = form_data.get('action')
    
    if action in ['chat', 'health', 'stats', 'clear_memory']:
        if action == 'chat':
            return await handle_php_chat(form_data)
        elif action == 'health':
            return await handle_php_health()
        elif action == 'stats':
            return await handle_php_stats()
        elif action == 'clear_memory':
            return await handle_php_clear_memory()
    
    # Fall back to PHP execution for other cases
    php_file_path = f"php_frontend/{path}"
    return await serve_php_file(php_file_path)

async def handle_php_chat(form_data):
    """Handle chat requests from PHP frontend."""
    message = form_data.get('message', '')
    session_id = form_data.get('session_id', 'php_session')
    
    if not message.strip():
        return Response(
            content=json.dumps({'error': 'Message cannot be empty'}),
            media_type="application/json"
        )
    
    try:
        # Process the question using the existing LLM manager
        result = bible_qa_system.llm_manager.process_question(message)
        
        if not result or "answer" not in result:
            return Response(
                content=json.dumps({'error': 'No response generated'}),
                media_type="application/json"
            )
        
        # Extract source documents if available
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": getattr(doc, 'metadata', {})
                })
        
        response_data = {
            "answer": result["answer"],
            "session_id": session_id,
            "sources": sources if sources else None
        }
        
        return Response(
            content=json.dumps(response_data),
            media_type="application/json"
        )
        
    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        return Response(
            content=json.dumps({'error': f'Error processing request: {str(e)}'}),
            media_type="application/json"
        )

async def handle_php_health():
    """Handle health check requests from PHP frontend."""
    try:
        vector_count = bible_qa_system.pinecone_manager.get_vector_count()
        response_data = {
            "status": "healthy",
            "message": "BibleBot API is running",
            "vector_count": vector_count
        }
        return Response(
            content=json.dumps(response_data),
            media_type="application/json"
        )
    except Exception as e:
        response_data = {
            "status": "error",
            "message": f"Service error: {str(e)}"
        }
        return Response(
            content=json.dumps(response_data),
            media_type="application/json"
        )

async def handle_php_stats():
    """Handle stats requests from PHP frontend."""
    try:
        stats = bible_qa_system.pinecone_manager.get_stats()
        if stats:
            response_data = {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_name": bible_qa_system.pinecone_manager.index_name
            }
            return Response(
                content=json.dumps(response_data),
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps({'error': 'Failed to retrieve stats'}),
                media_type="application/json"
            )
    except Exception as e:
        return Response(
            content=json.dumps({'error': f'Error getting stats: {str(e)}'}),
            media_type="application/json"
        )

async def handle_php_clear_memory():
    """Handle clear memory requests from PHP frontend."""
    try:
        success = bible_qa_system.llm_manager.clear_memory()
        if success:
            return Response(
                content=json.dumps({"message": "Memory cleared successfully"}),
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps({'error': 'Failed to clear memory'}),
                media_type="application/json"
            )
    except Exception as e:
        return Response(
            content=json.dumps({'error': f'Error clearing memory: {str(e)}'}),
            media_type="application/json"
        )

async def serve_php_file(php_file_path: str):
    """Execute PHP file and return the result."""
    try:
        # Check if file exists
        if not os.path.exists(php_file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Set up environment variables for PHP
        php_env = os.environ.copy()
        
        # Set FastAPI server URL for PHP to use
        php_env['API_BASE_URL'] = 'http://localhost:8000'
        
        # Execute PHP file
        result = subprocess.run(
            ["php", php_file_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            env=php_env
        )
        
        if result.returncode != 0:
            print(f"PHP Error: {result.stderr}")
            raise HTTPException(status_code=500, detail="PHP execution error")
        
        # Return the PHP output
        return Response(
            content=result.stdout,
            media_type="text/html"
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PHP interpreter not found")
    except Exception as e:
        print(f"Error serving PHP file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get Pinecone index statistics."""
    global bible_qa_system
    
    if bible_qa_system is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        stats = bible_qa_system.pinecone_manager.get_stats()
        if stats:
            return StatsResponse(
                total_vectors=stats.total_vector_count,
                dimension=stats.dimension,
                index_name=bible_qa_system.pinecone_manager.index_name
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve stats")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response."""
    global bible_qa_system
    
    if bible_qa_system is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Process the question using the existing LLM manager
        result = bible_qa_system.llm_manager.process_question(request.message)
        
        if not result or "answer" not in result:
            raise HTTPException(status_code=500, detail="No response generated")
        
        # Extract source documents if available
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": getattr(doc, 'metadata', {})
                })
        
        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id or "default",
            sources=sources if sources else None
        )
        
    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        if os.getenv("DEBUG", "").lower() == "true":
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/clear-memory")
async def clear_memory():
    """Clear the conversation memory."""
    global bible_qa_system
    
    if bible_qa_system is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        success = bible_qa_system.llm_manager.clear_memory()
        if success:
            return {"message": "Memory cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear memory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🌐 Starting BibleBot API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
