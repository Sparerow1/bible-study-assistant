"""
LangChain integration with Google Gemini AI model.
This file demonstrates various ways to use Gemini with LangChain.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.callbacks import StreamingStdOutCallbackHandler
import asyncio
import pinecone
from pinecone import Pinecone, ServerlessSpec


class GeminiLangChainBot:
    """
    A comprehensive LangChain wrapper for Google Gemini AI with Pinecone support.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", use_pinecone: bool = False):

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        self.use_pinecone = use_pinecone
        
        # Initialize the chat model
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize embeddings for vector operations
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        # Initialize Pinecone if requested
        if self.use_pinecone:
            self._setup_pinecone()
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def _setup_pinecone(self):
        """Setup Pinecone client and configuration."""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-gemini")
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Create index if it doesn't exist
        if self.pinecone_index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=768,  # Dimension for Google's embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.pinecone_environment
                )
            )
        
        self.pinecone_index = self.pc.Index(self.pinecone_index_name)
    
    def simple_chat(self, message: str) -> str:
        """
        Simple chat interaction with Gemini.
        
        Args:
            message: User message
            
        Returns:
            AI response
        """
        try:
            response = self.llm.invoke([HumanMessage(content=message)])
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_with_system_prompt(self, message: str, system_prompt: str) -> str:
        """
        Chat with a custom system prompt.
        
        Args:
            message: User message
            system_prompt: System prompt to guide the AI behavior
            
        Returns:
            AI response
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def conversation_chat(self, message: str) -> str:
        """
        Chat with conversation memory.
        
        Args:
            message: User message
            
        Returns:
            AI response with conversation context
        """
        try:
            response = self.conversation.predict(input=message)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_prompt_template_chain(self, template: str, input_variables: List[str]) -> LLMChain:
        """
        Create a chain with a custom prompt template.
        
        Args:
            template: Prompt template string
            input_variables: List of variable names in the template
            
        Returns:
            LLMChain with the custom template
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def setup_document_qa_chroma(self, documents_path: str) -> RetrievalQA:
        """
        Set up a document Q&A system using Chroma vector database.
        
        Args:
            documents_path: Path to directory containing text documents
            
        Returns:
            RetrievalQA chain for document question answering
        """
        # Load documents
        loader = DirectoryLoader(documents_path, glob="*.txt")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain
    
    def setup_document_qa_pinecone(self, documents_path: str, namespace: str = "") -> RetrievalQA:
        """
        Set up a document Q&A system using Pinecone vector database.
        
        Args:
            documents_path: Path to directory containing text documents
            namespace: Pinecone namespace for organizing vectors
            
        Returns:
            RetrievalQA chain for document question answering
        """
        if not self.use_pinecone:
            raise ValueError("Pinecone not initialized. Set use_pinecone=True when creating the bot.")
        
        # Load documents
        loader = DirectoryLoader(documents_path, glob="*.txt")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create Pinecone vector store
        vectorstore = PineconeVectorStore.from_documents(
            documents=texts,
            embedding=self.embeddings,
            index_name=self.pinecone_index_name,
            namespace=namespace
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain
    
    def add_documents_to_pinecone(self, documents_path: str, namespace: str = "") -> int:
        """
        Add documents to existing Pinecone index.
        
        Args:
            documents_path: Path to directory containing text documents
            namespace: Pinecone namespace for organizing vectors
            
        Returns:
            Number of document chunks added
        """
        if not self.use_pinecone:
            raise ValueError("Pinecone not initialized. Set use_pinecone=True when creating the bot.")
        
        # Load documents
        loader = DirectoryLoader(documents_path, glob="*.txt")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Add to existing Pinecone index
        vectorstore = PineconeVectorStore(
            index=self.pinecone_index,
            embedding=self.embeddings,
            namespace=namespace
        )
        
        # Add documents
        vectorstore.add_documents(texts)
        
        return len(texts)
    
    def search_pinecone(self, query: str, namespace: str = "", k: int = 3) -> List[Dict[str, Any]]:
        """
        Search Pinecone for similar documents.
        
        Args:
            query: Search query
            namespace: Pinecone namespace to search in
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        if not self.use_pinecone:
            raise ValueError("Pinecone not initialized. Set use_pinecone=True when creating the bot.")
        
        vectorstore = PineconeVectorStore(
            index=self.pinecone_index,
            embedding=self.embeddings,
            namespace=namespace
        )
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            })
        
        return formatted_results
    
    def delete_pinecone_namespace(self, namespace: str):
        """
        Delete all vectors in a Pinecone namespace.
        
        Args:
            namespace: Namespace to delete
        """
        if not self.use_pinecone:
            raise ValueError("Pinecone not initialized. Set use_pinecone=True when creating the bot.")
        
        self.pinecone_index.delete(delete_all=True, namespace=namespace)
    
    async def async_chat(self, message: str) -> str:
        """
        Asynchronous chat method.
        
        Args:
            message: User message
            
        Returns:
            AI response
        """
        try:
            response = await self.llm.ainvoke([HumanMessage(content=message)])
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of message dictionaries
        """
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "ai", "content": msg.content})
        return history


def example_usage():
    """
    Example usage of the GeminiLangChainBot.
    """
    # Initialize the bot (make sure to set GOOGLE_API_KEY environment variable)
    try:
        bot = GeminiLangChainBot()
        
        print("=== Simple Chat Example ===")
        response = bot.simple_chat("Hello! Tell me a joke about programming.")
        print(f"AI: {response}\n")
        
        print("=== Chat with System Prompt Example ===")
        system_prompt = "You are a helpful assistant that speaks like Shakespeare."
        response = bot.chat_with_system_prompt(
            "What is artificial intelligence?", 
            system_prompt
        )
        print(f"AI: {response}\n")
        
        print("=== Conversation with Memory Example ===")
        bot.conversation_chat("My name is John and I love pizza.")
        response = bot.conversation_chat("What do you know about me?")
        print(f"AI: {response}\n")
        
        print("=== Custom Prompt Template Example ===")
        template = """
        You are an expert {field} tutor. 
        Explain the concept of {concept} to a beginner in simple terms.
        
        Concept: {concept}
        Field: {field}
        
        Explanation:
        """
        
        chain = bot.create_prompt_template_chain(
            template, 
            ["field", "concept"]
        )
        
        response = chain.run(field="programming", concept="recursion")
        print(f"AI: {response}\n")
        
        print("=== Conversation History ===")
        history = bot.get_conversation_history()
        for entry in history:
            print(f"{entry['role'].title()}: {entry['content'][:100]}...")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your GOOGLE_API_KEY environment variable.")
    except Exception as e:
        print(f"Unexpected error: {e}")


async def async_example():
    """
    Example of asynchronous usage.
    """
    try:
        bot = GeminiLangChainBot()
        response = await bot.async_chat("What are the benefits of async programming?")
        print(f"Async AI: {response}")
    except Exception as e:
        print(f"Async error: {e}")


def pinecone_example():
    """
    Example usage with Pinecone vector database.
    """
    try:
        # Initialize bot with Pinecone support
        bot = GeminiLangChainBot(use_pinecone=True)
        
        print("=== Pinecone Vector Database Example ===")
        
        # Create a sample document directory (if you have documents)
        # qa_chain = bot.setup_document_qa_pinecone("./documents", namespace="bible")
        
        # Search example (if you have documents loaded)
        # results = bot.search_pinecone("What is love?", namespace="bible", k=3)
        # for i, result in enumerate(results, 1):
        #     print(f"Result {i} (Score: {result['similarity_score']:.3f}):")
        #     print(f"Content: {result['content'][:200]}...")
        #     print()
        
        print("Pinecone setup successful!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Set your API keys as environment variables
    # os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
    # os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"
    
    print("LangChain Gemini + Pinecone Integration Demo")
    print("=" * 50)
    
    # Run synchronous examples
    example_usage()
    
    # Run asynchronous example
    print("\n=== Async Example ===")
    asyncio.run(async_example())
    
    # Run Pinecone example
    print("\n=== Pinecone Example ===")
    pinecone_example()
