@echo off
cd .venv\Scripts
call activate.bat
cd ../..
cd Projects
cmd /k "prompt $G$S python $S"