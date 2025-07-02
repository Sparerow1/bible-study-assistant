from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_gemini import GeminiLangChainBot
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader  # Fixed import
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Fixed import
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import time

def upload_documents_in_batches(texts, embeddings, index_name, batch_size=50):
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
                    index_name=index_name,
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15, 
                                                        "score_threshold": 0.7})

    # 3. Conversational Retrieval Chain Implementation
    print("🤖 Initializing Gemini LLM...")
    bot = GeminiLangChainBot()  # Use your existing bot class properly
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"  # Important for ConversationalRetrievalChain
    )

    custom_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""You are a knowledgeable Bible scholar and teacher. Your role is to provide comprehensive, thoughtful, and spiritually enriching answers about the Bible and Christian faith.

Use the following Biblical passages and context to answer the question. Provide a detailed, well-structured response that includes:

1. Direct answer to the question
2. Relevant Bible verses and references
3. Historical and cultural context when applicable
4. Practical application or spiritual insight
5. Cross-references to related Biblical concepts

Biblical Context:
{context}

Previous Conversation:
{chat_history}

Question: {question}

Please provide a comprehensive answer that would be helpful for someone seeking to understand God's Word better. Include specific verse references and explain the meaning in a clear, accessible way.

Answer:"""
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=bot.llm,  # Use the LLM from your bot
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
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
                for i, doc in enumerate(sources[:2], 1):
                    content_preview = doc.page_content[:200].replace('\n', ' ').strip()
                    print(f"  {i}. {content_preview}...")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()