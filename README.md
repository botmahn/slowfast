# M2MVT x DAAD PySlowFast

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An unofficial pytorch implementation for [Early Anticipation of Driving Maneuvers](https://cvit.iiit.ac.in/research/projects/cvit-projects/daad) based on the [PySlowFast](https://github.com/facebookresearch/SlowFast) framework.

## Overview

This repository attempts to reproduce the methodologies and results from the paper [Early Anticipation of Driving Maneuvers](https://cvit.iiit.ac.in/research/projects/cvit-projects/daad) (Abdul Wasi et al., ECCV 2024). The implementation processes multi-view and multi-modal driving sequences using 6 different camera views.

### Key Features

- Specialized dataloader for multi-view (6 cameras) driving sequences.
- M2MVT architecture with spatio-temporal tubes, early fusion for the first 5 views and late fusion for the last (gaze) view.
- Additional learnable memory tokens.
- Built on Facebook AI Research's PySlowFast framework.

## Installation

Installation steps are taken from [here](https://github.com/facebookresearch/SlowFast/issues/743#issue-2750996881).

### Prerequisites

- Python 3.8
- CUDA 11.7
- PyTorch 1.13.0
- TorchVision 0.14.0 (compiled from source)

### Setup Environment

```bash
# Create and activate conda environment
conda create -n slowfast python=3.8
conda activate slowfast

# Install PyTorch ecosystem with CUDA support
conda install -y pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install FFmpeg
conda install -y -c conda-forge ffmpeg=4.2
```

### Rebuild TorchVision from Source

This step is necessary to fix video decoding issues:

```bash
# Uninstall current torchvision
pip uninstall -y torchvision

# Clone and build TorchVision v0.14.0
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.14.0
python setup.py install
cd ..
```

### Install Dependencies

```bash
# Install PyTorchVideo from source
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"

# Core dependencies
pip install simplejson opencv-python psutil
conda install -y -c conda-forge iopath
conda install -y tensorboard

# Analysis and data tools
pip install scikit-learn pandas
conda install -y -c conda-forge moviepy

# Additional frameworks
pip install 'git+https://github.com/facebookresearch/fairscale'
pip install cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install Detectron2
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

### Install PySlowFast

```bash
# Clone repository
git clone https://github.com/facebookresearch/slowfast

# Add to PYTHONPATH
echo 'export PYTHONPATH=/path/to/slowfast:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

# Build PySlowFast
cd slowfast
python setup.py build develop
```

## Fixing Known Issues

After installation, you'll need to apply these critical fixes:

### 1. Fix Import Paths in tools/run_net.py

Replace:
```python
# from vision.fair.slowfast.tools.demo_net import demo
# from vision.fair.slowfast.tools.test_net import test
# from vision.fair.slowfast.tools.train_net import train
# from vision.fair.slowfast.tools.visualization import visualize
```

With:
```python
from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize
```

### 2. Fix Import Paths in slowfast/utils/ava_eval_helper.py

Replace:
```python
# from vision.fair.slowfast.ava_evaluation import (
#     object_detection_evaluation,
#     standard_fields,
# )
```

With:
```python
from ava_evaluation import (
    object_detection_evaluation,
    standard_fields,
)
```

### 3. Fix "NumPy array not writable" Error in slowfast/datasets/decoder.py

Replace:
```python
# video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))
```

With:
```python
video_tensor = torch.from_numpy(np.frombuffer(np.array(video_handle), dtype=np.uint8))
```

## Using Custom Data

Using custom data is taken from [here](https://github.com/facebookresearch/SlowFast/issues/149#issuecomment-723265461).

Download the DAAD dataset from [here](https://cvit.iiit.ac.in/research/projects/cvit-projects/daad)

Follow these steps to prepare your custom dataset:

1. Create the following directory structure:
```
SlowFast/
├── configs/
│       └── MyData/
│               └── I3D_8x8_R50.yaml
├── data/
│       └── MyData/
│               ├── ClassA/
│               │      └── video1.mp4
│               ├── ClassB/
│               │      └── video2.mp4
│               ├── ClassC/
│               |      └── video3.mp4
│               ├── train.csv
│               ├── test.csv
│               ├── val.csv
│               └── classids.json
```

2. Create dataset handler:
   - Duplicate `slowfast/datasets/kinetics.py` and rename it to `mydata.py`
   - Replace all occurrences of "Kinetics" with "Mydata" (case-sensitive)
   - Add `from .mydata import Mydata` to `slowfast/datasets/__init__.py`

3. Create JSON class mapping file (`classids.json`):
```json
{"ClassA": 0, "ClassB": 1, "ClassC": 2}
```

4. Create CSV dataset split files with format:
```
/path/to/SlowFast/data/MyData/ClassA/video1.mp4 0
/path/to/SlowFast/data/MyData/ClassC/video3.mp4 2
```

5. Create configuration file by copying an existing one and changing "kinetics" to "mydata"

## Training and Testing

To train the model:

```bash
python tools/run_net.py --cfg configs/DAAD/MVITv2_S_16x4_daad.yaml >& ./logs/log_m2mvt_daadsixviews.txt &
```

To test the model:

```bash
vim configs/DAAD/MVITv2_S_16x4_daad.yaml

set TRAIN.ENABLE to False
set TEST.ENABLE to True
set NUM_GPUS = 1

RUN:
python tools/run_net.py --cfg configs/DAAD/MVITv2_S_16x4_daad.yaml
```

## Model Zoo

For pre-trained models and baseline results, refer to the PySlowFast [Model Zoo](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md). 

For M2MVT x DAAD weights, refer to our [MODEL ZOO](MODEL_ZOO.md).

## Acknowledgements

This work builds upon several important contributions:

- The PySlowFast framework developed by [Feichtenhofer et al.](https://github.com/facebookresearch/SlowFast)
- The driving anticipation methodologies presented in [DAAD](https://cvit.iiit.ac.in/research/projects/cvit-projects/daad)
- Fusion techniques explored in [Early or Late Fusion Matters: Efficient RGB-D Fusion in Vision Transformers for 3D Object Recognition](https://arxiv.org/pdf/2210.00843)
- Learnable Memory explored in [Fine-tuning Image Transformers using Learnable Memory](https://arxiv.org/abs/2203.15243) and [lucidrains implementation](https://github.com/lucidrains/vit-pytorch?tab=readme-ov-file#learnable-memory-vit)


```bibtex
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6202--6211},
  year={2019}
}

@inproceedings{adm2024daad,
  author       = {Abdul Wasi, Shankar Gangisetty, Shyam Nandan Rai and C. V. Jawahar},
  title        = {Early Anticipation of Driving Maneuvers},
  booktitle    = {ECCV (70)},
  series       = {Lecture Notes in Computer Science},
  volume       = {15128},
  pages        = {152--169},
  publisher    = {Springer},
  year         = {2024}
}

@misc{tziafas2023earlylatefusionmatters,
      title={Early or Late Fusion Matters: Efficient RGB-D Fusion in Vision Transformers for 3D Object Recognition}, 
      author={Georgios Tziafas and Hamidreza Kasaei},
      year={2023},
      eprint={2210.00843},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2210.00843}, 
}

@misc{sandler2022finetuningimagetransformersusing,
      title={Fine-tuning Image Transformers using Learnable Memory}, 
      author={Mark Sandler and Andrey Zhmoginov and Max Vladymyrov and Andrew Jackson},
      year={2022},
      eprint={2203.15243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2203.15243}, 
}

```

## License

This project is released under the [Apache 2.0 license](LICENSE), in accordance with the original PySlowFast repository.
