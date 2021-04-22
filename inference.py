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


def inference(image: str) -> str:
    """
    Returns image caption
    :param image_url: list of image URLs
    :return: Description to display.
    """

    translator = Translator()
    eng_caption = get_prediction([image])
    ru_caption = translator.translate(eng_caption, dest='ru').text

    return ru_caption