#!/bin/sh
# Справка: .wmz (gzip-WMF) → .svg и правка ссылок в .htm. Нужны libwmf-bin и gsfonts.
set -e
SRC="$1"
STAGE="$2"
rm -rf "$STAGE"
cp -r "$SRC" "$STAGE"
n=0
bad=0
for f in $(find "$STAGE" -name '*.wmz'); do
    svg="${f%.wmz}.svg"
    if gunzip -c "$f" > /tmp/_t.wmf 2>/dev/null && wmf2svg /tmp/_t.wmf -o "$svg" 2>/dev/null && [ -s "$svg" ]; then
        n=$((n + 1))
    else
        bad=$((bad + 1))
        [ -s "$svg" ] || printf '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20"><text x="2" y="14" font-size="12">?</text></svg>' > "$svg"
    fi
done
echo "converted: $n, failed: $bad"
python3 - "$STAGE" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
changed = 0
for p in root.rglob('*.htm*'):
    s = p.read_text(encoding='cp1251', errors='replace')
    if '.wmz' in s.lower():
        p.write_text(s.replace('.wmz', '.svg').replace('.WMZ', '.svg'), encoding='cp1251', errors='replace')
        changed += 1
print(f"rewritten htm: {changed}")
PY
