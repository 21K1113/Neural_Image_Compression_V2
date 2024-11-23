@echo off
cd .venv2\Scripts
call activate.bat
cd ../..
cd Projects
python image_compression.py IMAGE_PATH=data/Multilayer_para3_64.npy FP_G0_BITS=2 NUM_EPOCHS=320000 COMPRESSION_METHOD=4 IMAGE_DIMENSION=3 IMAGE_SIZE=64 CROP_MIP_LEVEL=5
cmd /k 