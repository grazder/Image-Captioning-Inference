# Image Captioning Inference

This project use models and weights from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
Bottom-up attention embeddings generated using [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention), which is pytorch implementation of [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).

## Requirements
- requirements.txt

## Install

You should install detectron with
```
python3 setup.py build develop
```

Also you should download weights for ResNet, Bottom-up attention models.

Or you can install and download models using `download.sh` script.

Repo installation code, which I use:
```
!git clone https://github.com/grazder/Image-Captioning-Inference.git
%cd Image-Captioning-Inference
!pip install -r requirements.txt
!bash download.sh
```
