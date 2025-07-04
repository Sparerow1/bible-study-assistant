# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Define the main analysis
a = Analysis(
    ['biblebot.py'],                    # Entry point script
    pathex=['.'],                       # Current directory
    binaries=[],
    datas=[
        ('bible_read.txt', '.'),        # Include Bible text file
        ('.env.example', '.'),          # Include example env file
        ('README.md', '.'),             # Include README
    ],
    hiddenimports=[
        'langchain_google_genai',
        'langchain_community',
        'langchain_text_splitters', 
        'langchain_pinecone',
        'pinecone',
        'chromadb',
        'tiktoken',
        'google.generativeai',
        'dotenv',
        'numpy',
        'pandas',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',                   # Exclude unnecessary packages
        'jupyter',
        'notebook',
        'pytest',
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
    upx=True,                          # Compress with UPX (optional)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,                      # Console application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',                   # Add icon if you have one
)
