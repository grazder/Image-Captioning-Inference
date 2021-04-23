from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from captioning import models
from captioning.utils import misc as utils
from captioning.utils.resnet_utils import myResnet
from captioning.utils import resnet
from captioning.data.dataloaderraw import *
from captioning.utils import eval_utils
from captioning.data.ImagePreprocessing import ImagePreprocessing

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

import argparse
from typing import List


class Captions:
    def __init__(self,
                 model_path: str = 'data/fc-resnet-weights/model.pth',
                 infos_path: str = 'data/fc-resnet-weights/infos.pkl',
                 model_type: str = 'resnet',
                 resnet_model_path: str = 'data/imagenet_weights/resnet101.pth',
                 bottom_up_model_path: str = 'data/bottom-up/faster_rcnn_from_caffe.pkl',
                 bottom_up_config_path: str = 'data/bottom-up/faster_rcnn_R_101_C4_caffe.yaml',
                 bottom_up_vocab: str = 'data/vocab/objects_vocab.txt',
                 device:str = 'cpu'):
        """
        Returns image caption
        :param model_path: path to model
        :param infos_path: path to model infos
        :param type: type of picture embedding - resnet or bottom-up
        :param resnet_model_path: path to resnet model
        :param bottom_up_model_path: path to bottom-up model
        :param bottom_up_config_path: path to bottom-up model config
        :param bottom_up_vocab: path to bottom-up model vocab
        :param device: device - cpu or gpu
        """

        self._model_path = model_path
        self._infos_path = infos_path

        assert model_type == 'resnet' or model_type == 'bottom-up'
        self._model_type = model_type

        self._resnet_model_path = resnet_model_path
        self._bottom_up_model_path = bottom_up_model_path
        self._bottom_up_config_path = bottom_up_config_path
        self._bottom_up_vocab = bottom_up_vocab
        self._device = device

        self.opt, self.infos, self.model = self._get_model()

        if self._model_type == 'resnet':
            preprocessor_model = self._get_resnet()
        elif self._model_type == 'bottom-up':
            preprocessor_model = self._get_bottom_up()

        self.preprocessor = ImagePreprocessing(preprocessor_model, self._model_type)

    def get_prediction(self, images: List[str]) -> List[str]:
        """
        Returns image caption
        :param images: list of image URLs
        :return: English image descriptions.
        """

        loader = DataLoaderRaw(self.preprocessor, images=images, batch_size=2)
        loader.dataset.ix_to_word = self.infos['vocab']

        split_predictions = eval_utils.eval_split(self.model, loader, vars(self.opt))

        return [x['caption'] for x in split_predictions]

    def _get_model(self):
        opt = argparse.Namespace(batch_size=0, beam_size=1, block_trigrams=0, coco_json='',
                                 decoding_constraint=0, diversity_lambda=0.5, dump_images=1,
                                 dump_json=1, dump_path=0, group_size=1, id='', image_folder='',
                                 image_root='', input_att_dir='', input_box_dir='', input_fc_dir='',
                                 input_json='', input_label_h5='', language_eval=0, length_penalty='',
                                 max_length=20, num_images=-1, remove_bad_endings=0, sample_method='greedy',
                                 split='test', suppress_UNK=1, temperature=1.0, verbose_beam=1, verbose_loss=0)

        opt.model = self._model_path
        opt.infos_path = self._infos_path
        opt.device = self._device
        opt.dataset = opt.input_json

        # Load infos
        with open(opt.infos_path, 'rb') as f:
            infos = utils.pickle_load(f)

        # override and collect parameters
        replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
        ignore = ['start_from']

        for k in vars(infos['opt']).keys():
            if k in replace:
                setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
            elif k not in ignore:
                if k not in vars(opt):
                    vars(opt).update({k: vars(infos['opt'])[k]})

        vocab = infos['vocab']

        opt.vocab = vocab
        model = models.setup(opt)
        del opt.vocab
        model.load_state_dict(torch.load(opt.model, map_location='cpu'))
        model.to(opt.device)
        model.eval()

        return opt, infos, model

    def _get_resnet(self):
        my_resnet = getattr(resnet, 'resnet101')()
        my_resnet.load_state_dict(torch.load(self._resnet_model_path))
        my_resnet = myResnet(my_resnet)
        my_resnet.eval()

        return my_resnet

    def _get_bottom_up(self):
        vg_classes = []
        with open(self._bottom_up_vocab) as f:
            for object in f.readlines():
                vg_classes.append(object.split(',')[0].lower().strip())

        MetadataCatalog.get("vg").thing_classes = vg_classes

        cfg = get_cfg()
        cfg.MODEL.DEVICE = self._device
        cfg.merge_from_file(self._bottom_up_config_path)
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        # VG Weight
        cfg.MODEL.WEIGHTS = self._bottom_up_model_path
        predictor = DefaultPredictor(cfg)

        return predictor
