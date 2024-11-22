@echo off
cd .venv2\Scripts
call activate.bat
cd ../..
tensorboard --logdir="tensorboard_show_log"
cmd /k