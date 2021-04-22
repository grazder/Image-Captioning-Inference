from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import skimage
import skimage.io

from ..data.ImagePreprocessing import ImagePreprocessing
from typing import List

class DataLoaderRaw:

    def __init__(self, images: List[str], embed_type: str = 'bottom-up', batch_size: int = 10):
        assert embed_type == 'resnet' or embed_type == 'bottom-up'

        self.files = images
        self.N = len(self.files)
        self.ids = [str(x) for x in range(self.N)]
        self.iterator = 0
        self.embed_type = embed_type

        self.dataset = self  # to fix the bug in eval
        self.preprocesser = ImagePreprocessing(self.embed_type)
        self.batch_size = batch_size

    def get_batch(self, batch_size = None):
        batch_size = batch_size or self.batch_size
        # if self.embed_type == 'resnet':
        #     fc_batch = np.ndarray((batch_size, 2048), dtype='float32')
        #     att_batch = np.ndarray((batch_size, 196, 2048), dtype='float32')
        # elif self.embed_type == 'bottom-up':
        #     fc_batch = np.ndarray((batch_size, 2048), dtype='float32')
        #     att_batch = np.ndarray((batch_size, 100, 2048), dtype='float32')

        fc_batch = []
        att_batch = []

        max_index = self.N
        infos = []

        for _ in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0

            self.iterator = ri_next

            img = skimage.io.imread(self.files[ri])
            tmp_fc, tmp_att = self.preprocesser.preprocess(img)

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            info_struct = {
                'id': self.ids[ri],
                'file_path': self.files[ri]
            }
            infos.append(info_struct)

        data = {
            'fc_feats': np.stack(fc_batch),
            'labels': np.zeros([batch_size, 0]),
            'masks': None,
            'bounds': {
                'it_pos_now': self.iterator,
                'it_max': self.N,
                'wrapped': False
            },
            'infos': infos}

        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        print(data['att_feats'].shape)
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0], ] = att_batch[i]

        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1

        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in data.items()}

        return data

    def reset_iterator(self):
        self.iterator = 0
