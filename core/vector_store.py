from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from config.config_setup_class import BibleQAConfig
from pathHandling.validation import Validation
import os
import sys
import time

class DocumentProcessor:
    """Handles document loading, splitting, and embedding operations."""
    
    def __init__(self, config: BibleQAConfig):
        self.config = config
        self.embeddings = None
        
    def initialize_embeddings(self, google_api_key: str) -> bool:
        """Initialize Google embeddings."""
        try:
            print("🔧 Initializing Google embeddings...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model,
                google_api_key=google_api_key
            )
            
            # Test the embeddings
            test_result = self.embeddings.embed_query("Test embedding")
            print(f"✅ Embeddings initialized! Dimension: {len(test_result)}")
            return True
            
        except Exception as e:
            print(f"❌ Embeddings initialization failed: {e}")
            print("💡 Please check your GOOGLE_API_KEY and internet connection.")
            return False
