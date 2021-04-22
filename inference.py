from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from captioning.data.dataloaderraw import *
from captioning.utils import eval_utils
import views

from typing import List
from googletrans import Translator


def get_prediction(images: List[str]) -> List[str]:
    """
    Returns image caption
    :param images: list of image URLs
    :return: English image descriptions.
    """

    loader = DataLoaderRaw(images, views.TYPE, 2)
    loader.dataset.ix_to_word = views.INFOS['vocab']

    split_predictions = eval_utils.eval_split(views.MODEL, loader, vars(views.OPT))

    return [x['caption'] for x in split_predictions]

if __name__ == '__main__':
    preds = get_prediction([
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRpAyuVYvA7Xp6WnXQlBVBMg',
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRI3WEb0IQDxkLE8MFwgdqOQ',
        # 'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRRDv-gy72oOxqHd8ZecLzuw',
        # 'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRujG9u4GLmZBwopXXDgfByQ',
        # 'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRvTAPzORkNwshNNSIoWQrmw',
    ])

    print(preds)