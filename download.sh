#!/bin/bash

python setup.py build develop

gdown --id 0B7fNdx_jAqhtSmdCNDVOVVdINWs -O resnet101.pth
mkdir ml/data/imagenet_weights/
mv resnet101.pth ml/data/imagenet_weights/

gdown --id 1QG4criqiVuCfur6juvH1ddi2twUS3bJ9 -O infos.pkl
mkdir ml/data/fc-resnet-weights/
mv infos.pkl ml/data/fc-resnet-weights/

gdown --id 1pSfTXtevxSh84LTsV-Ed-nolxPQBxFYQ -O model.pth
mkdir ml/data/fc-resnet-weights/
mv model.pth ml/data/fc-resnet-weights/

gdown --id 1tt6kaAQW6ZM0i7YcSJuIiAtZ2fbuU3H1 -O model.pth
mkdir ml/data/transformer-self-critical-weights/
mv model.pth ml/data/transformer-self-critical-weights/

gdown --id 1gvseH4bwghWzPrWfPMwumwbfvJvWYm9k -O infos.pkl
mkdir ml/data/transformer-self-critical-weights/
mv infos.pkl ml/data/transformer-self-critical-weights/

wget http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl 
mkdir ml/data/bottom-up/
mv faster_rcnn_from_caffe.pkl ml/data/bottom-up/