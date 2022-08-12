#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)" 
rm -rf env_mit
conda create --prefix env_mit/ python=3.9 -y
conda activate env_mit/
pip install --upgrade pip
pip install pyqt5 tqdm Pillow==9.2 opencv-python-headless