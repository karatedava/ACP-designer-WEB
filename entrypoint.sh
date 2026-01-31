#!/usr/bin/env bash
set -euo pipefail

echo "Checking / downloading model if needed..."
python download_models.py

echo "Starting the application..."
exec python app.py 