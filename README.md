

## Installation

```
pip install -r requirements.txt
```

Creating a virtual environment is recommended. Images are expected to be under "train" folder. Expected structure:

```
train/
    0/
    1/ 
    2/
```

## Usage

To train:

```
python train.py
```

To test video (if you want to test on your laptop's camera, give video_path as 0):

```
python test.py --video=<video_path> --model-path=<path_to_the_model>
```

To test an image:

```
python test.py --image=<image_path> --model-path=<path_to_the_model>
```