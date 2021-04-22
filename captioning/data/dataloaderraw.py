from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import skimage
import skimage.io

from ml.captioning.data.ImagePreprocessing import ImagePreprocessing
from typing import List

class DataLoaderRaw:

    def __init__(self, images: List[str], embed_type: str = 'bottom-up'):
        assert embed_type == 'resnet' or embed_type == 'bottom-up'

        self.files = images
        self.N = len(self.files)
        self.ids = [str(x) for x in range(self.N)]
        self.iterator = 0
        self.embed_type = embed_type

        self.dataset = self  # to fix the bug in eval
        self.preprocesser = ImagePreprocessing(self.embed_type)

    def get_batch(self, batch_size = 1):
        if self.embed_type == 'resnet':
            fc_batch = np.ndarray((batch_size, 2048), dtype='float32')
            att_batch = np.ndarray((batch_size, 196, 2048), dtype='float32')
        elif self.embed_type == 'bottom-up':
            pass

        max_index = self.N
        infos = []

        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0

            self.iterator = ri_next

            img = skimage.io.imread(self.files[ri])
            tmp_fc, tmp_att = self.preprocesser.preprocess(img)

            fc_batch[i] = tmp_fc
            att_batch[i] = tmp_att

            info_struct = {
                'id': self.ids[ri],
                'file_path': self.files[ri]
            }
            infos.append(info_struct)

        data = {
            'fc_feats': fc_batch,
            'att_feats': att_batch,
            'labels': np.zeros([batch_size, 0]),
            'masks': None,
            'att_masks': None,
            'bounds': {
                'it_pos_now': self.iterator,
                'it_max': self.N,
                'wrapped': False
            },
            'infos': infos}

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in data.items()}

        return data
