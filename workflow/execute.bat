@echo off
title Video KB Validator

echo.
echo ========================================
echo Starting Video KB Validator...
echo ========================================
echo.

cd /d "C:\Users\eiealxi\OneDrive - Ericsson\Desktop\workflow"

call ".\.venv\Scripts\activate.bat"

python ".\SW\demo.py"

echo.
echo ========================================
echo Program exited
echo ========================================
pause