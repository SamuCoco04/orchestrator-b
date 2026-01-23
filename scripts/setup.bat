@echo off
setlocal

if not exist .venv (
  python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install -e .
python -m src.main --mode mock --brief brief.md
