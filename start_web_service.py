#!/usr/bin/env python3
"""
BibleBot Web Service Startup Script

This script starts the BibleBot as a FastAPI web service.
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the BibleBot web service."""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Add the script directory to Python path
    sys.path.insert(0, str(script_dir))
    
    # Set environment variables if not already set
    if not os.getenv("PORT"):
        os.environ["PORT"] = "8000"
    
    if not os.getenv("HOST"):
        os.environ["HOST"] = "0.0.0.0"
    
    # Get configuration from environment
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print("🚀 Starting BibleBot Web Service...")
    print(f"📡 Server will be available at: http://{host}:{port}")
    print(f"🌐 Web interface: http://{host}:{port}/web")
    print(f"📚 API documentation: http://{host}:{port}/docs")
    print(f"🔧 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print("=" * 60)
    
    try:
        # Start the FastAPI server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 BibleBot Web Service stopped.")
    except Exception as e:
        print(f"❌ Error starting BibleBot Web Service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
