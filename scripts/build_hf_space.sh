#!/usr/bin/env bash
# Assemble the Hugging Face Space (Docker SDK) bundle in dist/hf-space/.
# Only the app, the small artifacts and the backtest CSVs: no raw data, no secrets.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=dist/hf-space
rm -rf "$OUT"
mkdir -p "$OUT/reports"

cp Dockerfile .dockerignore requirements-deploy.txt "$OUT/"
cp deploy/hf_space/README.md "$OUT/README.md"
cp -R src app artifacts .streamlit "$OUT/"
cp -R reports/stage4 "$OUT/reports/stage4"
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$OUT" -name ".DS_Store" -delete

# Safety checks before anything is uploaded.
if find "$OUT" -name ".env" | grep -q .; then echo "refusing: .env in bundle" >&2; exit 1; fi
if grep -rIl "AIza[0-9A-Za-z_-]\{20,\}" "$OUT" >/dev/null 2>&1; then
  echo "refusing: something that looks like a Google API key is in the bundle" >&2; exit 1
fi
big=$(find "$OUT" -type f -size +10M)
if [ -n "$big" ]; then echo "warning: files over 10 MB (need Xet/LFS): $big" >&2; fi

echo "Bundle ready in $OUT ($(du -sh "$OUT" | cut -f1)):"
(cd "$OUT" && find . -maxdepth 2 -not -path "*/src/*" -not -path "*/app/*" | sort | sed 's/^/  /')
