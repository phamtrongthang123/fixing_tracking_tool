
## Quickstart

### Dependencies
Run

```bash
conda create --prefix env_mit/ python=3.9 -y
conda activate env_mit/
pip install --upgrade pip
pip install pyqt5 tqdm Pillow==9.2 opencv-python-headless
```


### Run
Given the zip folder B5KplpNzL-A.zip and video B5KplpNzL-A.mp4: 
1. Create `./videos` folder and `./frames` folder.
2. Run `python split_frame.py` to split all the videos in `./videos` into frames
3. Run `python anno.py B5KplpNzL-A 3` to start the script. 

## The annotation tool 

### Command

The command is `python anno.py <video_index> <max-people>`. By default, if you don't pass the `max-people` argument, the program will find the max number of people in the videos.

### Features

This annotation tool has these features:
- Can move backward forward between frames.
- Automatically saving the progress after updating.
- Can resume the current progress. 
- You can set the maximum number of people in the video. With this, the tool will update all the "larger than max" index. For example, if the output label is 26, but we only have 3 people, and the label's bounding box belongs to 2, then after you fill the box at 26 as 2, all the frames that have 26 in it will replace 26 with 2. 

The demo: https://drive.google.com/file/d/18_dl4wjeq6ibh-fBmSEkjJpD3uVMFO94/view?usp=sharing

