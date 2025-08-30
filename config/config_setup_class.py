import os
from pathHandling.validation import Validation

class BibleQAConfig:
    """Configuration class for Bible Q&A system"""
    
    def __init__(self):
        self.embedding_dimension = 768
        self.retriever_k = 30
        self.llm_temperature = 0.7
        self.llm_model = "gemini-2.0-flash-lite"
        self.embedding_model = "models/embedding-001"
        
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        config = cls()
        # Override defaults with environment variables if they exist
        config.retriever_k = int(os.getenv("RETRIEVER_K", config.retriever_k))
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
        data_file = Validation.get_resource_path(filepath)
        if not os.path.exists(data_file):
            print(f"❌ Error: {data_file} not found!")
            print("💡 Please ensure the Bible text file is in the current directory.")
            print(f"   Expected location: {os.path.abspath(filepath)}")
            return False
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                f.read(100)  # Try to read first 100 chars
            print(f"✅ Bible text file found and readable: {data_file}")
            return True
        except UnicodeDecodeError:
            print(f"❌ File encoding error: {filepath}")
            print("💡 Please ensure the file is saved in UTF-8 encoding.")
            return False
        except Exception as e:
            print(f"❌ Error reading file {filepath}: {e}")
            return False
