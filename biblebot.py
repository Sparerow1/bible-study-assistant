from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import sys
from typing import Optional, List, Dict, Any
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
import traceback


class BibleBotError(Exception):
    """Custom exception for Bible Bot errors."""
    pass


class EnvironmentValidator:
    """Validates environment variables and setup."""
    
    REQUIRED_VARS = [
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX_NAME"
    ]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in cls.REQUIRED_VARS if not os.getenv(var)]
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"   • {var}")
            print("\n💡 Please check your .env file and ensure all variables are set.")
            return False
        
        print("✅ All environment variables validated!")
        return True


class DocumentManager:
    """Handles document loading, processing, and validation."""
    
    def __init__(self, file_path: str = "./bible_read.txt"):
        self.file_path = file_path
        self.documents = None
        self.text_chunks = None
    
    def validate_file(self) -> bool:
        """Check if the Bible text file exists and is readable."""
        if not os.path.exists(self.file_path):
            print(f"❌ Error: {self.file_path} not found!")
            print("💡 Please ensure the Bible text file is in the current directory.")
            print(f"   Expected location: {os.path.abspath(self.file_path)}")
            return False
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                f.read(100)
            print(f"✅ Bible text file found and readable: {self.file_path}")
            return True
        except UnicodeDecodeError:
            print(f"❌ File encoding error: {self.file_path}")
            print("💡 Please ensure the file is saved in UTF-8 encoding.")
            return False
        except Exception as e:
            print(f"❌ Error reading file {self.file_path}: {e}")
            return False
    
    def load_documents(self) -> bool:
        """Load documents from the file."""
        try:
            print("📖 Loading Bible text...")
            loader = TextLoader(self.file_path)
            self.documents = loader.load()
            
            if not self.documents:
                raise BibleBotError("No documents loaded!")
            
            file_size = os.path.getsize(self.file_path)
            print(f"📁 File size: {file_size / 1024 / 1024:.1f} MB")
            print(f"📄 Loaded {len(self.documents)} document(s)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return False
    
    def split_documents(self, chunk_size: int = 1500, chunk_overlap: int = 300) -> bool:
        """Split documents into chunks."""
        if not self.documents:
            print("❌ No documents to split! Load documents first.")
            return False
        
        try:
            print("📝 Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            self.text_chunks = text_splitter.split_documents(self.documents)
            
            if not self.text_chunks:
                raise BibleBotError("No text chunks created!")
            
            print(f"✅ Created {len(self.text_chunks)} text chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error splitting documents: {e}")
            return False


class PineconeManager:
    """Manages Pinecone vector database operations."""
    
    def __init__(self, api_key: str, index_name: str, dimension: int = 768):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.pinecone_client = None
        self.index = None
        self.vectorstore = None
    
    def initialize(self) -> bool:
        """Initialize Pinecone client and index."""
        try:
            print("🔧 Initializing Pinecone...")
            self.pinecone_client = Pinecone(api_key=self.api_key)
            print("✅ Pinecone client initialized!")
            
            return self._setup_index()
            
        except Exception as e:
            print(f"❌ Pinecone initialization failed: {e}")
            return False
    
    def _setup_index(self) -> bool:
        """Setup or connect to Pinecone index."""
        try:
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            print(f"📋 Found {len(existing_indexes)} existing indexes")
            
            if self.index_name not in existing_indexes:
                print(f"📊 Creating new Pinecone index: {self.index_name}")
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print("✅ Index created successfully!")
            else:
                print(f"✅ Using existing index: {self.index_name}")
            
            self.index = self.pinecone_client.Index(self.index_name)
            stats = self.index.describe_index_stats()
            print(f"📊 Index stats: {stats.total_vector_count} vectors")
            return True
            
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                print(f"✅ Index '{self.index_name}' already exists, continuing...")
                self.index = self.pinecone_client.Index(self.index_name)
                return True
            else:
                print(f"❌ Error setting up index: {e}")
                return False
    
    def get_vector_count(self) -> int:
        """Get current vector count in the index."""
        if not self.index:
            return 0
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            print(f"❌ Error getting vector count: {e}")
            return 0
    
    def upload_documents_in_batches(self, texts: List, embeddings, batch_size: int = 30) -> bool:
        """Upload documents to Pinecone in smaller batches to avoid size limits."""
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"📊 Uploading {len(texts)} documents in {total_batches} batches of {batch_size}...")
        
        vectorstore = None
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"⏳ Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            try:
                if vectorstore is None:
                    # First batch - create the vectorstore
                    vectorstore = PineconeVectorStore.from_documents(
                        batch,
                        embeddings,
                        index_name=self.index_name,
                    )
                else:
                    # Subsequent batches - add to existing vectorstore
                    vectorstore.add_documents(batch)
                
                print(f"✅ Batch {batch_num} uploaded successfully!")
                
                # Small delay to avoid rate limiting
                if batch_num < total_batches:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Error uploading batch {batch_num}: {e}")
                return None
        
        print("🎉 All documents uploaded successfully!")
        return vectorstore

def main():   
    load_dotenv()  # Load environment variables from .env file
    
    # Load environment variables and set up Pinecone
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    embedding_dimension = 768 
    index_name = os.getenv("PINECONE_INDEX_NAME", "biblebot-index")
    
    # Check if index exists, and create if not
    existing_indexes = [index.name for index in pinecone.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"📊 Creating new Pinecone index: {index_name}")
        try:
            pinecone.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("✅ Index created successfully!")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                print(f"✅ Index '{index_name}' already exists, continuing...")
            else:
                print(f"❌ Error creating index: {e}")
                return
    else:
        print(f"✅ Using existing index: {index_name}")
    
    # Connect to the Pinecone index
    index = pinecone.Index(index_name)

    # Check if bible_read.txt exists
    if not os.path.exists("./bible_read.txt"):
        print("❌ Error: bible_read.txt not found!")
        print("Please make sure the Bible text file is in the current directory.")
        return

    # 1. Document Loading and Processing
    print("📖 Loading Bible text...")
    loader = TextLoader("./bible_read.txt")
    documents = loader.load()
    
    # Get file size for reference
    file_size = os.path.getsize("./bible_read.txt")
    print(f"📁 File size: {file_size / 1024 / 1024:.1f} MB")

    print(f"📝 Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Smaller chunks to reduce batch size
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ Created {len(texts)} text chunks")

    # Initialize embeddings
    print("🔧 Initializing Google embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    print("✅ Embeddings initialized!")

    # 2. Check if vectors already exist in Pinecone
    index_stats = index.describe_index_stats()
    vector_count = index_stats.total_vector_count
    
    if vector_count == 0:
        # Upload documents in batches
        vectorstore = upload_documents_in_batches(texts, embeddings, index_name, batch_size=30)
        if vectorstore is None:
            print("❌ Failed to upload documents!")
            return
    else:
        print(f"✅ Found {vector_count} existing vectors in Pinecone")
        # Create vectorstore connection to existing data
        vectorstore = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings
        )

    # Create a retriever from the Pinecone index
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15},
                                         search_type="similarity",)

    # 3. Conversational Retrieval Chain Implementation
    print("🤖 Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        streaming=True,
        convert_system_message_to_human=True,
        model="gemini-2.5-flash",  # Use your existing bot's model
        temperature=0.7, 
        callbacks=[StreamingStdOutCallbackHandler()])
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"  # Important for ConversationalRetrievalChain
    )

    custom_prompt = PromptTemplate(
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

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Use the LLM from your bot
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )

    # 4. Interactive Bible Q&A
    print("\n" + "="*50)
    print("🔍 Bible Q&A System Ready!")
    print("Ask questions about the Bible. Commands:")
    print("  'exit' - Quit the program")
    print("  'clear' - Clear conversation memory")
    print("  'stats' - Show Pinecone index statistics")
    print("="*50)

    while True:
        try:
            query = input("\n📖 Your question: ").strip()
            
            if query.lower() == "exit":
                print("👋 God bless!")
                break
            elif query.lower() == "clear":
                memory.clear()
                print("🧹 Memory cleared!")
                continue
            elif query.lower() == "stats":
                stats = index.describe_index_stats()
                print(f"📊 Index stats: {stats.total_vector_count} vectors, {stats.dimension} dimensions")
                continue
            elif not query:
                continue

            print("🤖 AI: ", end="", flush=True)
            result = qa_chain({"question": query})
            
            print(result["answer"])
            
            # Show source documents
            if "source_documents" in result and result["source_documents"]:
                sources = result["source_documents"]
                print(f"\n📚 Biblical References ({len(sources)}):")
                for i, doc in enumerate(sources, 1):
                    content_preview = doc.page_content
                    print(f"  {i}. {content_preview}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()