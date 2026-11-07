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
        done = subprocess.run([sys.executable, str(HERE / name)], capture_output=True, text=True)
        tail = [line for line in done.stdout.strip().splitlines() if line.strip()][-1:]
        print(f"  {'ok' if done.returncode == 0 else 'ПРОВАЛ'}: {tail[0] if tail else 'нет вывода'}")
        if done.returncode != 0:
            failed.append(name)
            print(done.stderr.strip()[-800:])
    print(f"\nпровалено: {len(failed)} из {len(SCRIPTS)}" if failed else f"\nвсе {len(SCRIPTS)} проверок прошли")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
