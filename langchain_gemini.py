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
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.callbacks import StreamingStdOutCallbackHandler
import asyncio
from pinecone import Pinecone, ServerlessSpec

class GeminiLangChainBot:
    """
    A comprehensive LangChain wrapper for Google Gemini AI.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        
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
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
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
    
    def setup_document_qa(self, documents_path: str) -> RetrievalQA:
        """
        Set up a document Q&A system using vector embeddings.
        
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


if __name__ == "__main__":
    # Set your Google API key here or as an environment variable
    # os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
    
    print("LangChain Gemini Integration Demo")
    print("=" * 50)
    
    # Run synchronous examples
    example_usage()
    
    # Run asynchronous example
    print("\n=== Async Example ===")
    asyncio.run(async_example())
