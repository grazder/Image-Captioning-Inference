from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Model import Model


if __name__ == '__main__':
    model = Model(model_path='data/fc-resnet-weights/',
                  type='resnet',
                  resnet_model_path='data/imagenet_weights/resnet101.pth',
                  bottom_up_model_path='data/bottom-up/faster_rcnn_from_caffe.pkl',
                  bottom_up_config_path='data/bottom-up/faster_rcnn_R_101_C4_caffe.yaml',
                  bottom_up_vocab='data/vocab/objects_vocab.txt',
                  device='cpu')

    preds = model.get_prediction([
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRpAyuVYvA7Xp6WnXQlBVBMg',
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRI3WEb0IQDxkLE8MFwgdqOQ',
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRRDv-gy72oOxqHd8ZecLzuw',
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRujG9u4GLmZBwopXXDgfByQ',
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRvTAPzORkNwshNNSIoWQrmw',
    ])

    print(preds)
