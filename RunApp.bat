@echo off
REM ─── Activate the venv ──────────────────────────
call "%~dp0.venv\Scripts\activate.bat"

REM ─── Install any missing deps ───────────────────
pip install --upgrade pip
pip install -r "%~dp0requirements.txt"

REM ─── Launch Streamlit ───────────────────────────
streamlit run "%~dp0app\app.py"

pause
