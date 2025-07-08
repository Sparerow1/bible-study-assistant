import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables."""
    test_vars = {
        'GOOGLE_API_KEY': 'test_google_key',
        'PINECONE_API_KEY': 'test_pinecone_key',
        'PINECONE_INDEX_NAME': 'test-index',
        'PINECONE_ENVIRONMENT': 'test-env'
    }
    
    with patch.dict(os.environ, test_vars):
        yield test_vars

@pytest.fixture
def mock_bible_file(tmp_path):
    """Create a temporary Bible file for testing."""
    content = """Genesis 1:1 In the beginning God created the heavens and the earth.
Genesis 1:2 Now the earth was formless and empty, darkness was over the surface of the deep."""
    
    bible_file = tmp_path / "test_bible.txt"
    bible_file.write_text(content)
    return str(bible_file)

@pytest.fixture
def mock_pinecone():
    """Mock Pinecone client."""
    with patch('pinecone.Pinecone') as mock:
        yield mock

@pytest.fixture
def mock_google_genai():
    """Mock Google Generative AI."""
    with patch('google.generativeai.configure') as mock:
        yield mock