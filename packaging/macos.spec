# PyInstaller: .app для macOS. Справка кладётся внутрь пакета, математика — в wheel-пакете.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent          # корень репозитория
datas = [(str(ROOT / "kuotik_port" / "assets" / "help.zip"), "kuotik_port/assets")]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=collect_submodules("kuotik_port"),
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="kuotik",
          console=False, argv_emulation=False)      # нативная арка: universal2 требует fat-бинарников от всех колёс
coll = COLLECT(exe, a.binaries, a.datas, name="kuotik")
if sys.platform == "darwin":                    # BUNDLE — только macOS, на Linux проверяется Analysis
    app = BUNDLE(coll, name="КУ ОТИК.app",
                 bundle_identifier="dev.wlix13.kuotik",
                 info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "12.0"})
