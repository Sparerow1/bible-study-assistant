

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_gemini import GeminiLangChainBot
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader  # Example document loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#  Load environment variables
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# 1. Document Loading and Processing
loader = TextLoader("./bible_read.txt")  # Load your document
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-001",  # Specify the embedding model
    google_api_key=os.getenv("GOOGLE_API_KEY")  # Load API key from environment
)

# 2. Vector Store Setup
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever()

# 3.  Conversational Retrieval Chain Implementation
llm = GeminiLangChainBot()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
)

# 4. Interact with the chatbot
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    result = qa_chain({"question": query, "chat_history": chat_history})
    print("AI:", result["answer"])
    chat_history.append((query, result["answer"]))