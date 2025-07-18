import os
import time
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


# --- Configuration ---
# The name of the Pinecone index to create/use.
# Pinecone index names must be all lowercase, alphanumeric, and use '-' for hyphens.
INDEX_NAME = "biblebot-index" 
# The path to the text file to upload.
SOURCE_FILE_PATH = "content.txt" # specify the path of the text file
# The number of documents to upload in a single batch.
BATCH_SIZE = 100

def main():
    """
    Main function to load data, create embeddings, and upload to Pinecone.
    """
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    # 1. Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # 2. Create the index if it doesn't exist
    # Google's embedding-001 model produces 768-dimensional vectors.
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Index created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    index = pc.Index(INDEX_NAME)

    # 3. Load the document
    print(f"Loading document from {SOURCE_FILE_PATH}...")
    loader = TextLoader(SOURCE_FILE_PATH, encoding='utf-8')
    documents = loader.load()

    # 4. Split the document into smaller chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Document split into {len(docs)} chunks.")

    # 5. Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    # 6. Upload the documents and their embeddings to Pinecone in batches
    print(f"Uploading {len(docs)} chunks to Pinecone index '{INDEX_NAME}' in batches of {BATCH_SIZE}...")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        print(f"Uploading batch {i//BATCH_SIZE + 1}...")
        vector_store.add_documents(batch)
        time.sleep(1) # Wait for 1 second between batches to avoid rate limiting

    print("Upload complete.")

if __name__ == "__main__":

    if not os.path.exists(SOURCE_FILE_PATH):
        print(f"Error: The file '{SOURCE_FILE_PATH}' was not found.")
        print("Please create the file and add your content to it.")
    else:
        main()