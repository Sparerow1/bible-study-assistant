# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[('data/bible_read.txt', 'data'), ('.env', '.')],
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
        'chromadb',
        'google.generativeai',
        'dotenv'
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
