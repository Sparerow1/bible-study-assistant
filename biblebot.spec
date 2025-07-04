# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

block_cipher = None

# Get current directory and find the correct script path
current_dir = Path.cwd()

# Try to find the main script in different possible locations
possible_scripts = [
    'src/biblebot.py',
    'biblebot.py', 
    'src/qa_package/qa_class.py',
    'qa_package/qa_class.py'
]

main_script = None
for script in possible_scripts:
    if (current_dir / script).exists():
        main_script = script
        break

if main_script is None:
    raise FileNotFoundError("Could not find main script. Please ensure one of these files exists: " + str(possible_scripts))

print(f"Using main script: {main_script}")

# Prepare data files - only include existing files
datas_list = []

# Check and add data files
data_files = [
    ('src/bible_read.txt', '.'),
    ('bible_read.txt', '.'),
    ('.env.example', '.'),
    ('README.md', '.'),
]

for src, dst in data_files:
    if (current_dir / src).exists():
        datas_list.append((src, dst))
        print(f"Including data file: {src}")

# Check and add Python packages
package_dirs = [
    ('src/config_package', 'config_package/'),
    ('src/llm_package', 'llm_package/'),
    ('src/vector_store_package', 'vector_store_package/'),
    ('src/qa_package', 'qa_package/'),
]

for src, dst in package_dirs:
    if (current_dir / src).exists():
        datas_list.append((src, dst))
        print(f"Including package: {src}")

print(f"Total data items to include: {len(datas_list)}")

# Define the main analysis
a = Analysis(
    [main_script],                     # Use the found script
    pathex=[
        str(current_dir),              # Current directory
        str(current_dir / 'src'),      # src directory
    ],
    binaries=[],
    datas=datas_list,                  # Use the pre-filtered list
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
        
        # Your custom modules (conditional based on what exists)
        'config_package',
        'config_package.config_setup_class',
        'config_package.setup_pinecone',
        'llm_package',
        'llm_package.llm_manager', 
        'vector_store_package',
        'vector_store_package.vector_store',
        'qa_package',
        'qa_package.qa_class',
        
        # Additional imports that might be needed
        'src.config_package',
        'src.llm_package',
        'src.vector_store_package', 
        'src.qa_package',
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
        'test',
        'tests',
        'unittest',
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
    upx=False,                         # Disable UPX compression (can cause issues on Linux/WSL)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,                      # Console application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Remove icon and version for Linux/WSL compatibility
)
