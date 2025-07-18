import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import time

# Load environment variables from .env file

# --- Configuration ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_DIRECTORY = "content/pdf_files"
PINECONE_INDEX_NAME = "bible-index-pdf" # The number of documents to upload in a single batch.
BATCH_SIZE = 100
EMBEDDING_DIMENSION = 768 
EMBEDDING_MODEL = "models/embedding-001"

def setup_pinecone():
    """Initializes Pinecone and creates the index if it doesn't exist."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Index created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
        
    return pc.Index(PINECONE_INDEX_NAME)

def process_and_upload():
    """Processes PDFs and uploads them to Pinecone in batches."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    index = setup_pinecone()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"Error: Directory not found at '{PDF_DIRECTORY}'")
        return

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIRECTORY}'.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    
    batch_to_upload = []
    total_vectors_uploaded = 0

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            file_path = os.path.join(PDF_DIRECTORY, pdf_file)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            chunks = text_splitter.split_documents(documents)
            
            # Get all chunk texts from the current PDF
            chunk_texts = [chunk.page_content for chunk in chunks]
            
            # Embed all chunks for the current document in a single batch call
            chunk_embeddings = embeddings.embed_documents(chunk_texts)

            # Ensure we have a matching number of chunks and embeddings
            if len(chunks) != len(chunk_embeddings):
                print(f"\nWarning: Mismatch in chunks and embeddings for {pdf_file}. Skipping this file.")
                continue

            for i, chunk in enumerate(chunks):
                vector_data = {
                    "id": f"{pdf_file}-{total_vectors_uploaded + len(batch_to_upload)}",
                    "values": chunk_embeddings[i],
                    "metadata": {
                        "source": pdf_file,
                        "text": chunk.page_content,
                        "page": chunk.metadata.get('page', 0)
                    }
                }
                batch_to_upload.append(vector_data)
                
                if len(batch_to_upload) >= BATCH_SIZE:
                    index.upsert(vectors=batch_to_upload)
                    total_vectors_uploaded += len(batch_to_upload)
                    print(f"\nUploaded batch of {len(batch_to_upload)}. Total uploaded: {total_vectors_uploaded}")
                    batch_to_upload = []

        except Exception as e:
            print(f"\nError processing file {pdf_file}: {e}")

    # Upload any remaining vectors
    if batch_to_upload:
        index.upsert(vectors=batch_to_upload)
        total_vectors_uploaded += len(batch_to_upload)
        print(f"\nUploaded final batch of {len(batch_to_upload)}. Total uploaded: {total_vectors_uploaded}")

    print("\n--- Upload Complete ---")
    print(f"Total vectors in index: {index.describe_index_stats()['total_vector_count']}")

if __name__ == "__main__":
    process_and_upload()