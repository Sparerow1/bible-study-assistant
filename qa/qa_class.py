import os
import sys
import traceback
from typing import List
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from config.config_setup_class import BibleQAConfig, EnvironmentValidator
from config.setup_pinecone import PineconeManager
from core.vector_store import DocumentProcessor
from llm.llm_manager import LLMManager
from pathHandling.validation import Validation


class BibleQASystem:
    """Main Bible Q&A system orchestrating all components."""
    
    def __init__(self):
        self.config = BibleQAConfig.from_env()
        self.pinecone_manager = None
        self.document_processor = None
        self.llm_manager = None
        self.vectorstore = None
        self.retriever = None
        
    def initialize(self) -> bool:
        """Initialize all system components."""
        # Attempt to load .env from bundled location
        env_path = Validation.load_env_from_bundle()
        if env_path:
            load_dotenv(env_path)
            print("✅ Environment variables loaded from bundled .env file")
        else:
            print("⚠️  No .env file found, using system environment variables")
            load_dotenv()  # Fallback to default behavior
        
        print("🚀 Starting Bible Q&A System...")
        print("=" * 50)
        
        # Validate environment
        if not EnvironmentValidator.validate_environment():
            return False
        
        if not EnvironmentValidator.validate_file(self.config.bible_file_path):
            return False
        
        # Initialize components
        if not self._initialize_pinecone():
            return False
            
        if not self._initialize_document_processor():
            return False
            
        if not self._setup_vectorstore():
            return False
            
        if not self._initialize_llm():
            return False
            
        print("\n🎉 Bible Q&A System initialized successfully!")
        return True
    
    def _initialize_pinecone(self) -> bool:
        """Initialize Pinecone manager."""
        self.pinecone_manager = PineconeManager(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding_dimension=self.config.embedding_dimension
        )
        return self.pinecone_manager.initialize()
    
    def _initialize_document_processor(self) -> bool:
        """Initialize document processor with embeddings."""
        self.document_processor = DocumentProcessor(self.config)
        return self.document_processor.initialize_embeddings(os.getenv("GOOGLE_API_KEY"))
    
    def _setup_vectorstore(self) -> bool:
        """Setup vectorstore with documents."""
        vector_count = self.pinecone_manager.get_vector_count()
        
        if vector_count == 0:
            print("📊 No existing vectors found. Loading and uploading documents...")
            texts = self.document_processor.load_and_split_documents(self.config.bible_file_path)
            if not texts:
                return False
                
            self.vectorstore = self.document_processor.upload_documents_in_batches(
                texts, self.pinecone_manager.index_name
            )
            if not self.vectorstore:
                return False
        else:
            print(f"✅ Found {vector_count} existing vectors in Pinecone")
            self.vectorstore = PineconeVectorStore(
                index_name=self.pinecone_manager.index_name,
                embedding=self.document_processor.embeddings
            )
        
        # Create retriever
        try:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retriever_k},
                search_type="similarity"
            )
            print("✅ Retriever created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating retriever: {e}")
            return False
    
    def _initialize_llm(self) -> bool:
        """Initialize LLM manager and create QA chain."""
        self.llm_manager = LLMManager(self.config)
        
        if not self.llm_manager.initialize_llm(os.getenv("GOOGLE_API_KEY")):
            return False
            
        if not self.llm_manager.initialize_memory():
            return False
            
        return self.llm_manager.create_qa_chain(self.retriever)
    
    def run_interactive_session(self):
        """Run the interactive Q&A session."""
        print("\n" + "="*60)
        print("🔍 Bible Q&A System Ready!")
        print("Ask questions about the Bible. Commands:")
        print("  'exit' - Quit the program")
        print("  'clear' - Clear conversation memory")
        print("  'stats' - Show Pinecone index statistics")
        print("  'help' - Show help message")
        print("="*60)
        
        while True:
            try:
                query = input("\n📖 Your question: ").strip()
                
                if not self._handle_command(query):
                    continue
                    
                if not query:
                    continue
                
                self._process_user_question(query)
            except KeyboardInterrupt:
                print("\n👋 May God bless your continued study of His Word!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if os.getenv("DEBUG", "").lower() == "true":
                    traceback.print_exc()
                continue
    
    def _handle_command(self, query: str) -> bool:
        """Handle special commands. Returns False if command was processed."""
        if query.lower() == "exit":
            print("👋 God bless your continued study of His Word!")
            sys.exit(0)
            
        elif query.lower() == "clear":
            if self.llm_manager.clear_memory():
                print("🧹 Memory cleared!")
            else:
                print("❌ Error clearing memory")
            return False
            
        elif query.lower() == "stats":
            stats = self.pinecone_manager.get_stats()
            if stats:
                print(f"📊 Index stats: {stats.total_vector_count} vectors, {stats.dimension} dimensions")
            else:
                print("❌ Error getting stats")
            return False
            
        elif query.lower() == "help":
            self._show_help()
            return False
            
        return True
    
    def _show_help(self):
        """Show help message."""
        print("\n💡 Available commands:")
        print("  'exit' - Quit the program")
        print("  'clear' - Clear conversation memory")
        print("  'stats' - Show Pinecone index statistics")
        print("  'help' - Show this help message")
        print("\n📖 Example questions:")
        print("  • 什么是爱？")
        print("  • 圣经如何教导我们祷告？")
        print("  • 耶稣的比喻有什么意义？")
        print("  • 如何理解救恩？")
    
    def _process_user_question(self, query: str):
        """Process a user question and display results."""
        print("🔍 Searching Scripture and preparing answer...")
        
        result = self.llm_manager.process_question(query)
        if not result:
            print("❌ No answer received from the system")
            return
        
        if "answer" in result and result["answer"]:
            print("\n" + "="*60)
            print("📖 Biblical Answer:")
            print("="*60)
            print(result["answer"])
        else:
            print("❌ Empty response from AI")
            return
        
        # self._display_source_documents(result.get("source_documents", []))
    
    def _display_source_documents(self, sources: List):
        """Display source documents with proper formatting."""
        if not sources:
            print("\n⚠️  No source documents were retrieved for this question.")
            return
        
        print(f"\n📚 Biblical References ({len(sources)} documents):")
        print("=" * 60)
        
        for i, doc in enumerate(sources, 1):
            try:
                print(f"\n📖 Source {i}:")
                print("-" * 30)
                content = doc.page_content.strip()
                print(content[100:200] + ("..." if len(content) > 200 else ""))
                
                # Show metadata if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"📋 Metadata: {doc.metadata}")
                    
            except Exception as e:
                print(f"❌ Error displaying source {i}: {e}")
        
        print("=" * 60)

