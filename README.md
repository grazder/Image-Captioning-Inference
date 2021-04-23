# Image Captioning Inference

This project use models and weights from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
Bottom-up attention embeddings generated using [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention), which is pytorch implementation of [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).

![](/readme_pics/example.png)

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
git clone https://github.com/grazder/Image-Captioning-Inference.git
cd Image-Captioning-Inference
pip install -r requirements.txt
bash download.sh
```

## Models

There are a lot of models from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). Which you can find in [MODEL_ZOO](https://github.com/ruotianluo/self-critical.pytorch/blob/master/MODEL_ZOO.md).

## Object initialization and usage example
```
from Captions import Captions
import os

model_fc_resnet = Captions(
                  model_path='data/fc-resnet-weights/model.pth',
                  infos_path='data/fc-resnet-weights/infos.pkl',
                  model_type='resnet',
                  resnet_model_path='data/imagenet_weights/resnet101.pth',
                  bottom_up_model_path='data/bottom-up/faster_rcnn_from_caffe.pkl',
                  bottom_up_config_path='data/bottom-up/faster_rcnn_R_101_C4_caffe.yaml',
                  bottom_up_vocab='data/vocab/objects_vocab.txt',
                  device='cpu'
                  )

images = os.listdir('example_images/')
paths = [os.path.join('example_images', x) for x in images]

preds = model_fc_resnet.get_prediction(paths)

for i, pred in enumerate(preds):
    print(f'{images[i]}: {pred}')
```


