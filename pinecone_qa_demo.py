

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_gemini import GeminiLangChainBot
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader  # Example document loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#  Load environment variables and set up Pinecone
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

embedding_dimension = 768 
index_name = os.getenv("PINECONE_INDEX_NAME", "biblebot-index")
# Check if index exists, and create if not
if index_name not in pinecone.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",  # Or other appropriate metric
    )
# Connect to the Pinecone index
index = pinecone.Index(index_name)


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