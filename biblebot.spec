# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

block_cipher = None

# Get current directory
current_dir = Path.cwd()

# Define the main analysis
a = Analysis(
    ['src/biblebot.py'],               # Updated path if using new structure
    pathex=[str(current_dir)],         # Current directory
    binaries=[],
    datas=[
        # Include data files
        ('data/bible_read.txt', 'data/'),        # Bible text file
        ('.env.example', '.'),                   # Example env file
        ('README.md', '.'),                      # README
        # Include all Python packages from src
        ('src/config_package', 'config_package/'),
        ('src/llm_package', 'llm_package/'),
        ('src/vector_store_package', 'vector_store_package/'),
        ('src/qa_package', 'qa_package/'),
    ],
    hiddenimports=[
        # Core LangChain imports
        'langchain',
        'langchain_google_genai',
        'langchain_community',
        'langchain_community.document_loaders',
        'langchain_text_splitters',
        'langchain_pinecone',
        'langchain.chains',
        'langchain.memory',
        'langchain.prompts',
        'langchain.callbacks',
        
        # Vector stores and embeddings
        'pinecone',
        'chromadb',
        'tiktoken',
        
        # Google AI
        'google.generativeai',
        'google.ai.generativelanguage',
        'google.auth',
        
        # Other dependencies
        'dotenv',
        'numpy',
        'pandas',
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
        
        # Your custom modules
        'config_package',
        'config_package.config_setup_class',
        'config_package.setup_pinecone',
        'llm_package',
        'llm_package.llm_manager',
        'vector_store_package',
        'vector_store_package.vector_store',
        'qa_package',
        'qa_package.qa_class',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'matplotlib.pyplot',
        'jupyter',
        'notebook',
        'pytest',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'IPython',
        'sphinx',
        'setuptools',
        'pip',
        'wheel',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Create the PYZ archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BibleBot',                    # Executable name
    debug=False,                        # Set to True for debugging
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                          # Compress with UPX (reduces size)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,                      # Console application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
    version='version_info.txt' if os.path.exists('version_info.txt') else None,
)
