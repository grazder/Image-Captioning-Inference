from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from captioning import models
import argparse
from captioning.utils import misc as utils
import torch
from captioning.utils.resnet_utils import myResnet
from captioning.utils import resnet

import os

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

TYPE = 'bottom-up'

if TYPE == 'bottom-up':
    model_path = 'data/transformer-self-critical-weights/'
elif TYPE == 'resnet':
    model_path = 'data/fc-resnet-weights/'


def get_model(model_path):
    opt = argparse.Namespace(batch_size=0, beam_size=1, block_trigrams=0, coco_json='',
                             decoding_constraint=0, diversity_lambda=0.5, dump_images=1,
                             dump_json=1, dump_path=0, group_size=1, id='', image_folder='',
                             image_root='', input_att_dir='', input_box_dir='', input_fc_dir='',
                             input_json='', input_label_h5='', language_eval=0, length_penalty='',
                             max_length=20, num_images=-1, remove_bad_endings=0, sample_method='greedy',
                             split='test', suppress_UNK=1, temperature=1.0, verbose_beam=1, verbose_loss=0)

    opt.model = model_path + 'model.pth'
    opt.infos_path = model_path + 'infos.pkl'
    opt.device = 'cpu'
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
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    vocab = infos['vocab']  # ix -> word mapping

    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab
    model.load_state_dict(torch.load(opt.model, map_location='cpu'))
    model.to(opt.device)
    model.eval()

    return opt, infos, model


def get_resnet():
    cnn_model = 'resnet101'
    my_resnet = getattr(resnet, cnn_model)()
    my_resnet.load_state_dict(torch.load('data/imagenet_weights/' + cnn_model + '.pth'))
    my_resnet = myResnet(my_resnet)
    my_resnet.eval()

    return my_resnet

def get_bottom_up():
    data_path = 'data/vocab/'

    vg_classes = []
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())

    MetadataCatalog.get("vg").thing_classes = vg_classes

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file("data/bottom-up/faster_rcnn_R_101_C4_caffe.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "data/bottom-up/faster_rcnn_from_caffe.pkl"
    predictor = DefaultPredictor(cfg)

    return predictor


OPT, INFOS, MODEL = get_model(model_path)
MY_RESNET = get_resnet()
MY_BOTTOM_UP = get_bottom_up()
