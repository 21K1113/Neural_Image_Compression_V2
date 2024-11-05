@echo off
cd .venv\Scripts
call activate.bat
cd ../..
cd Projects
python sample22-3-2.py num_bits=2 num_epochs=160000
python sample22-3-2.py num_bits=4 num_epochs=160000
python sample22-3-2.py num_bits=8 num_epochs=160000
python sample22-2.py num_bits=2 num_epochs=160000
python sample22-2.py num_bits=4 num_epochs=160000
python sample22-2.py num_bits=8 num_epochs=160000
python sample22-3.py num_bits=2 num_epochs=160000
python sample22-3.py num_bits=4 num_epochs=160000
python sample22-3.py num_bits=8 num_epochs=160000
cmd /k 