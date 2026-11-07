"""Точка входа для PyInstaller: абсолютный импорт, относительный в замороженном скрипте не работает."""

from kuotik_port.app import main

raise SystemExit(main())
