from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from config.config_setup_class import BibleQAConfig
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
    
    def get_resource_path(relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
            print(f"Running from PyInstaller temp dir: {base_path}")
        except Exception:
            base_path = os.path.abspath(".")
            print(f"Running from source: {base_path}")

        full_path = os.path.join(base_path, relative_path)
        print(f"Looking for data file at: {full_path}")
        return full_path

    def load_and_split_documents(self, filepath: str) -> Optional[List]:
        """Load and split documents from file."""
        try:
            print("📖 Loading Bible text...")
            loader = TextLoader(filepath)
            documents = loader.load()
            
            if not documents:
                print("❌ No documents loaded!")
                return None
            
            # Get file info
            file_size = os.path.getsize(filepath)
            print(f"📁 File size: {file_size / 1024 / 1024:.1f} MB")
            print(f"📄 Loaded {len(documents)} document(s)")
            
            print("📝 Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.text_separators
            )
            
            texts = text_splitter.split_documents(documents)
            
            if not texts:
                print("❌ No text chunks created!")
                return None
            
            print(f"✅ Created {len(texts)} text chunks")
            return texts
            
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return None
        except Exception as e:
            print(f"❌ Error loading/splitting documents: {e}")
            return None
    
    def upload_documents_in_batches(self, texts: List, index_name: str) -> Optional[PineconeVectorStore]:
        """Upload documents to Pinecone in batches with error handling."""
        if not texts or not self.embeddings:
            print("❌ No text chunks or embeddings not initialized!")
            return None
        
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size
        print(f"📊 Uploading {len(texts)} documents in {total_batches} batches of {self.config.batch_size}...")
        
        vectorstore = None
        successful_batches = 0
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_num = (i // self.config.batch_size) + 1
            
            print(f"⏳ Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            if self._upload_batch(batch, index_name, batch_num, vectorstore):
                successful_batches += 1
                if vectorstore is None:  # First successful batch creates vectorstore
                    vectorstore = PineconeVectorStore(
                        index_name=index_name,
                        embedding=self.embeddings
                    )
            
            # Delay between batches
            if batch_num < total_batches:
                time.sleep(1)
        
        if successful_batches > 0:
            print(f"🎉 Upload completed! {successful_batches}/{total_batches} batches successful")
            return vectorstore
        else:
            print("❌ No batches were successfully uploaded!")
            return None
    
    def _upload_batch(self, batch: List, index_name: str, batch_num: int, 
                     existing_vectorstore: Optional[PineconeVectorStore]) -> bool:
        """Upload a single batch with retry logic."""
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                if existing_vectorstore is None:
                    # First batch - create the vectorstore
                    PineconeVectorStore.from_documents(
                        batch, self.embeddings, index_name=index_name
                    )
                else:
                    # Subsequent batches - add to existing vectorstore
                    existing_vectorstore.add_documents(batch)
                
                print(f"✅ Batch {batch_num} uploaded successfully!")
                return True
                
            except Exception as e:
                print(f"⚠️  Batch {batch_num} failed (attempt {retry + 1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Batch {batch_num} failed after {max_retries} attempts")
        
        return False
