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
python classifier.py --train
```

To test:

```
python classifier.py --test --model-path=<path_to_the_model>
```

To video test:

```
python classifier.py --video --model-path=<path_to_the_model>
```