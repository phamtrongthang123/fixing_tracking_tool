from pathlib import Path 

VIDEO = Path("videos")
FRAMES = Path("frames")

import os 
from tqdm import tqdm 
loader = tqdm(sorted(list(VIDEO.glob("*"))))
os.makedirs(FRAMES, exist_ok=True)
for vid in loader: 
    saved = FRAMES/f"{vid.name.split('.')[0]}"
    os.makedirs(saved, exist_ok=True)
    os.system(f"ffmpeg -i {vid} {saved}/%10d.png")