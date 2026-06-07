@echo off
set PYTHONPATH=%cd%\src
python scripts\build_all.py
python -m uvicorn --app-dir src artrec.api.main:app --host 0.0.0.0 --port 8000
