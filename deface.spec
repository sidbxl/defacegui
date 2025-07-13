# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- Add data files ---
# Add the onnx model
datas = [('deface/centerface.onnx', 'deface')]

# Add data files for onnxruntime providers and cv2
datas += collect_data_files('onnxruntime.providers')
datas += collect_data_files('cv2')


# --- Hidden Imports ---
# Collect all submodules for onnxruntime
hiddenimports = collect_submodules('onnxruntime')

# Add other necessary hidden imports
hiddenimports += [
    # imageio plugins
    'imageio.plugins.ffmpeg',
    'imageio.v3',
    # For PyQt5
    'PyQt5.sip',
    'PyQt5.QtNetwork',
]


a = Analysis(
    ['deface/deface_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='defacegui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # This creates a windowed executable
    icon='readme-img/defaceGUI-icon.png'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='defacegui'
)
