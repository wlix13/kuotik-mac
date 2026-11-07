from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["check_app.py", "check_lab2.py", "check_lab3.py", "check_lab4.py",
           "check_lab5.py", "check_lab6.py", "check_lab7.py", "check_lab8.py"]


def main() -> int:
    failed = []
    for name in SCRIPTS:
        print(f"— {name}")
        done = subprocess.run([sys.executable, str(HERE / name)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        out = (done.stdout or "").strip()
        tail = [line for line in out.splitlines() if line.strip()][-1:]
        print(f"  {'ok' if done.returncode == 0 else 'ПРОВАЛ'}: {tail[0] if tail else 'нет вывода'}")
        if done.returncode != 0:
            failed.append(name)
            print((done.stderr or "").strip()[-800:])
    print(f"\nпровалено: {len(failed)} из {len(SCRIPTS)}" if failed else f"\nвсе {len(SCRIPTS)} проверок прошли")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
