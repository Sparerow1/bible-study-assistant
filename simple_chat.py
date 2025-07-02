"""
Simple example script demonstrating basic Gemini usage with LangChain.
"""

from langchain_gemini import GeminiLangChainBot
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Simple demonstration of Gemini with LangChain.
    """
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set your GOOGLE_API_KEY in the .env file or as an environment variable.")
        return
    
    try:
        # Initialize the bot with Pinecone support
        bot = GeminiLangChainBot(use_pinecone=True)
        
        # Setup Q&A system with documents
        documents_path = "./documents"
        qa_chain = None
        
        # Check if documents folder exists, create sample if not
        if not os.path.exists(documents_path):
            print("📁 Creating sample documents folder...")
            os.makedirs(documents_path, exist_ok=True)
            
            # Create sample document
            with open(f"{documents_path}/sample.txt", "w") as f:
                f.write("""
LangChain is a framework for developing applications powered by language models.
It provides tools for prompt management, chains, memory, and agents.

Pinecone is a vector database that enables similarity search and is commonly used
for retrieval-augmented generation (RAG) applications.

Vector databases store embeddings of text and allow for semantic search.
This enables finding relevant documents based on meaning rather than exact keywords.

Together, LangChain and Pinecone can build powerful question-answering systems
that can search through large document collections to find relevant information.
""")
            print("✅ Sample document created!")
        
        # Setup Q&A system
        try:
            print("📚 Setting up Pinecone Q&A system...")
            qa_chain = bot.setup_document_qa_pinecone(documents_path, namespace="chat_qa")
            print("✅ Q&A system ready!")
        except Exception as e:
            print(f"⚠️  Q&A setup failed: {e}")
            print("Falling back to regular chat mode...")
        
        print("Gemini LangChain Bot initialized successfully!")
        if qa_chain:
            print("🔍 Q&A mode: Ask questions about your documents")
        print("Type 'quit' to exit, 'clear' to clear memory, 'chat' for regular chat, 'qa' for Q&A mode\n")
        
        mode = "qa" if qa_chain else "chat"
        
        while True:
            prompt_prefix = "🔍 Q&A" if mode == "qa" else "💬 Chat"
            user_input = input(f"[{prompt_prefix}] You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                bot.clear_memory()
                print("Memory cleared!")
                continue
            elif user_input.lower() == 'chat':
                mode = "chat"
                print("Switched to chat mode")
                continue
            elif user_input.lower() == 'qa':
                if qa_chain:
                    mode = "qa"
                    print("Switched to Q&A mode")
                else:
                    print("Q&A system not available")
                continue
            elif not user_input:
                continue
            
            print("AI: ", end="", flush=True)
            
            if mode == "qa" and qa_chain:
                # Q&A mode - search documents and answer
                try:
                    result = qa_chain({"query": user_input})
                    answer = result["result"]
                    sources = result.get("source_documents", [])
                    
                    print(answer)
                    
                    if sources:
                        print(f"\n📚 Sources:")
                        for i, doc in enumerate(sources, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            print(f"  {i}. {os.path.basename(source)}")
                    
                except Exception as e:
                    print(f"Error in Q&A: {e}")
            else:
                # Regular chat mode
                response = bot.conversation_chat(user_input)
                print(response)
            
            print()  # New line after response
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
