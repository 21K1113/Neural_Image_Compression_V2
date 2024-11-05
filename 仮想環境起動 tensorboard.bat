@echo off
cd .venv\Scripts
call activate.bat
cd ../..
tensorboard --logdir="log/sample22-3-2_cuda_Multilayer_para3_64.npy_32_True_160000_8"
cmd /k