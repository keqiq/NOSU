# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui/app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Include all files from the gui directory
        ('gui/*.py', 'gui'),
        # Include files from the Model folder
        ('Model/OSUModel.py', 'Model'),
        ('Model/modelmanager/KeypressModel.py', 'Model/modelmanager'),
        ('Model/modelmanager/PositionModel.py', 'Model/modelmanager'),
        ('Model/modelmanager/ModelManager.py', 'Model/modelmanager'),
        # Include all .py files from the utils folder
        ('utils/*.py', 'utils'),
        # Other files
        ('_internal/replay_template.osr', '.'),
        ('_internal/config.json', '.'),

    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules
        'torchvision',
        'torchaudio',
        'numpy.tests',
        'scipy.tests',  
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NOSU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NOSU', 
)