#!/usr/bin/env sh
# Запуск без Docker под Linux/macOS: заглушки в фоне, приложение — в этом терминале.
# Перед первым запуском: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

"$PY" -m uvicorn mock_ml.app:app --app-dir mocks --port 8001 &
ML_PID=$!
"$PY" -m uvicorn mock_toir.app:app --app-dir mocks --port 8002 &
TOIR_PID=$!
trap 'kill $ML_PID $TOIR_PID 2>/dev/null' EXIT INT TERM

"$PY" -m uvicorn app.main:app --port 8000 --reload
