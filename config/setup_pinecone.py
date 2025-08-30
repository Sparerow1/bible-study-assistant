from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from typing import Optional, List, Dict, Any


class PineconeManager:
    """Manages Pinecone vector database operations."""
    
    def __init__(self, api_key: str, index_name: str, embedding_dimension: int = 768):
        self.api_key = api_key
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.pinecone_client = None
        self.index = None
    
    def initialize(self) -> bool:
        """Initialize Pinecone client and index."""
        try:
            print("🔧 Initializing Pinecone...")
            self.pinecone_client = Pinecone(api_key=self.api_key)
            print("✅ Pinecone client initialized!")
            
            if not self._ensure_index_exists():
                return False
                
            self.index = self.pinecone_client.Index(self.index_name)
            
            # Test the connection
            stats = self.index.describe_index_stats()
            print(f"📊 Index stats: {stats.total_vector_count} vectors")
            return True
            
        except Exception as e:
            print(f"❌ Pinecone initialization failed: {e}")
            print("💡 Please check your PINECONE_API_KEY and network connection.")
            return False
    
    def _ensure_index_exists(self) -> bool:
        """Ensure the Pinecone index exists, create if necessary."""
        try:
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            print(f"📋 Found {len(existing_indexes)} existing indexes")
            
            if self.index_name not in existing_indexes:
                print(f"📊 Creating new Pinecone index: {self.index_name}")
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    cloud="aws",
                    region="us-east-1"
                )
                print("✅ Index created successfully!")
            else:
                print(f"✅ Using existing index: {self.index_name}")
            
            return True
            
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                print(f"✅ Index '{self.index_name}' already exists, continuing...")
                return True
            else:
                print(f"❌ Error creating index: {e}")
                return False
    
    def get_vector_count(self) -> int:
        """Get the current vector count in the index."""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            print(f"❌ Error getting vector count: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed index statistics."""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}
