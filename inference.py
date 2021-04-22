from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ml.captioning.data.dataloaderraw import *
from ml.captioning.utils import eval_utils
from ml import views

from typing import List
from googletrans import Translator


def get_prediction(images: List[str]) -> List[str]:
    """
    Returns image caption
    :param images: list of image URLs
    :return: English image descriptions.
    """

    loader = DataLoaderRaw(images, views.TYPE)
    loader.dataset.ix_to_word = views.INFOS['vocab']

    split_predictions = eval_utils.eval_split(views.MODEL, loader, vars(views.OPT))

    return split_predictions['caption']

if __name__ == '__main__':
    preds = get_prediction([
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRpAyuVYvA7Xp6WnXQlBVBMg',
        'https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRI3WEb0IQDxkLE8MFwgdqOQ'
    ])