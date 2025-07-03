from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import time
from typing import Optional, List, Dict, Any
import traceback


class BibleQAConfig:
    """Configuration class for Bible Q&A system."""
    
    def __init__(self):
        self.embedding_dimension = 768
        self.chunk_size = 1500
        self.chunk_overlap = 300
        self.batch_size = 30
        self.retriever_k = 15
        self.llm_temperature = 0.7
        self.llm_model = "gemini-2.5-flash"
        self.embedding_model = "models/embedding-001"
        self.bible_file_path = "./bible_read.txt"
        self.text_separators = ["\n\n", "\n", ". ", " ", ""]
        
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        config = cls()
        # Override defaults with environment variables if they exist
        config.chunk_size = int(os.getenv("CHUNK_SIZE", config.chunk_size))
        config.batch_size = int(os.getenv("BATCH_SIZE", config.batch_size))
        config.retriever_k = int(os.getenv("RETRIEVER_K", config.retriever_k))
        config.bible_file_path = os.getenv("BIBLE_FILE_PATH", config.bible_file_path)
        return config


class EnvironmentValidator:
    """Validates environment variables and file dependencies."""
    
    REQUIRED_ENV_VARS = [
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX_NAME"
    ]
    
    @classmethod
    def validate_environment(cls) -> bool:
        """Validate that all required environment variables are set."""
        missing_vars = []
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"   • {var}")
            print("\n💡 Please check your .env file and ensure all variables are set.")
            return False
        
        print("✅ All environment variables validated!")
        return True
    
    @classmethod
    def validate_file(cls, filepath: str) -> bool:
        """Check if the required Bible text file exists and is readable."""
        if not os.path.exists(filepath):
            print(f"❌ Error: {filepath} not found!")
            print("💡 Please ensure the Bible text file is in the current directory.")
            print(f"   Expected location: {os.path.abspath(filepath)}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                f.read(100)  # Try to read first 100 chars
            print(f"✅ Bible text file found and readable: {filepath}")
            return True
        except UnicodeDecodeError:
            print(f"❌ File encoding error: {filepath}")
            print("💡 Please ensure the file is saved in UTF-8 encoding.")
            return False
        except Exception as e:
            print(f"❌ Error reading file {filepath}: {e}")
            return False


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
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
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


class LLMManager:
    """Manages the Language Model and conversation chain."""
    
    def __init__(self, config: BibleQAConfig):
        self.config = config
        self.llm = None
        self.memory = None
        self.qa_chain = None
        
    def initialize_llm(self, google_api_key: str) -> bool:
        """Initialize Gemini LLM."""
        try:
            print("🤖 Initializing Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=google_api_key,
                streaming=True,
                convert_system_message_to_human=True,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            
            # Test the LLM
            test_response = self.llm.invoke("Say 'Hello' in Chinese")
            print(f"✅ LLM initialized! Test response: {test_response.content[:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            print("💡 Please check your GOOGLE_API_KEY and model availability.")
            return False
    
    def initialize_memory(self) -> bool:
        """Initialize conversation memory."""
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            print("✅ Memory initialized")
            return True
        except Exception as e:
            print(f"❌ Error initializing memory: {e}")
            return False
    
    def create_qa_chain(self, retriever) -> bool:
        """Create the conversational retrieval chain."""
        try:
            custom_prompt = self._create_pastoral_prompt()
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                verbose=True,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
            )
            print("✅ Q&A chain created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating Q&A chain: {e}")
            return False
    
    def _create_pastoral_prompt(self) -> PromptTemplate:
        """Create the pastoral guidance prompt template."""
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""你是一位博学、幽默、且富有同情心的圣经学者和属灵导师，同时也是一位经验丰富的牧者。你的角色是提供全面、深思熟虑且具有属灵启发的圣经和基督教信仰答案，并以牧者的心肠来关怀和指导询问者。

作为一位好牧者，请遵循以下原则：

牧者品格与策略：
• 以基督的爱心和耐心回应每一个问题
• 倾听并理解询问者内心的需要和挣扎
• 提供既有真理又有恩典的平衡回答
• 用温柔而坚定的语气传达神的话语
• 避免论断，而是以慈爱引导人回到神面前
• 为询问者的属灵成长和实际需要祷告考虑
• 提供既有深度又容易理解的解释

请使用以下圣经经文和背景来回答问题，且不要超出以下经文和背景所提供的范围回答，如果不知道，就回答不知道。请提供详细且结构清晰的回答，包括：

1. 对问题的直接回答（以牧者的关怀开始）
2. 相关的圣经经文和引用（解释其中的属灵含义）
3. 历史和文化背景（如适用）
4. 实际应用或属灵见解（针对现代信徒的生活）
5. 与相关圣经概念的交叉引用
6. 牧者的鼓励和祷告方向（如适用）

圣经背景：
{context}

之前的对话：
{chat_history}

问题：{question}

作为一位属灵导师，请提供一个全面的答案，对于那些寻求更好理解神话语的人有帮助。请包含具体的经文引用，并以清晰易懂的方式解释含义。在回答中体现牧者的关怀、智慧和属灵的深度。请用中文回答。

如果遇到与圣经不相关的问题，请以牧者的心肠温柔地将话题引导回到圣经相关的问题，同时关心询问者的属灵需要。避免使用个人观点或未经证实的解释，但要以牧者的经验和智慧来应用圣经真理。

记住，你不仅是在传授知识，更是在牧养灵魂，帮助人们在基督里成长。

回复："""
        )
    
    def process_question(self, question: str) -> Optional[Dict[str, Any]]:
        """Process a question through the QA chain."""
        if not self.qa_chain:
            print("❌ QA chain not initialized!")
            return None
            
        try:
            result = self.qa_chain({"question": question})
            return result
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return None
    
    def clear_memory(self) -> bool:
        """Clear conversation memory."""
        try:
            if self.memory:
                self.memory.clear()
                return True
            return False
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")
            return False


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
        load_dotenv()
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
        
        self._display_source_documents(result.get("source_documents", []))
    
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
                print(content)
                
                # Show metadata if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"📋 Metadata: {doc.metadata}")
                    
            except Exception as e:
                print(f"❌ Error displaying source {i}: {e}")
        
        print("=" * 60)


def main():
    """Main entry point."""
    try:
        bible_qa = BibleQASystem()
        
        if bible_qa.initialize():
            bible_qa.run_interactive_session()
            return 0
        else:
            print("❌ Failed to initialize Bible Q&A system")
            return 1
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        if os.getenv("DEBUG", "").lower() == "true":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)