# -*- mode: python ; coding: utf-8 -*-
import os
import pinecone

block_cipher = None

# Get the actual pinecone package path
pinecone_path = os.path.dirname(pinecone.__file__)
version_file = os.path.join(/home/yc19a/miniconda3/lib/python3.13/site-packages/pinecone, '__version__')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('data/bible_read.txt', 'data'), 
        ('.env', '.'),
        # Add Pinecone version file using actual path
        (version_file, 'pinecone/'),
    ],
    hiddenimports=[
        'qa',
        'qa.qa_class',
        'config',
        'config.config_setup_class',
        'config.setup_pinecone',
        'core',
        'core.vector_store',
        'llm',
        'llm.llm_manager',
        'langchain',
        'langchain_google_genai',
        'langchain_pinecone',
        'langchain_community',
        'langchain_text_splitters',
        'pinecone',
        'pinecone.core',
        'pinecone.utils',
        'pinecone.utils.version',
        'chromadb',
        'google.generativeai',
        'dotenv',
        # Additional hidden imports for dependencies
        'urllib3',
        'requests',
        'certifi',
        'typing_extensions',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BibleBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
