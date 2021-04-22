from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from . import misc as utils

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def eval_split(model, loader, eval_kwargs={}):
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)
    device = eval_kwargs.get('device', 'cuda')

    # Make sure in the evaluation mode
    model.eval()

    predictions = []

    data = loader.get_batch()

    tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
    tmp = [_.to(device) if _ is not None else _ for _ in tmp]
    fc_feats, att_feats, labels, masks, att_masks = tmp

    # forward the model to also get generated samples for each image
    with torch.no_grad():
        tmp_eval_kwargs = eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 1})
        seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        seq = seq.data

    sents = utils.decode_sequence(model.vocab, seq)

    for k, sent in enumerate(sents):
        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
        if eval_kwargs.get('dump_path', 0) == 1:
            entry['file_name'] = data['infos'][k]['file_path']
        predictions.append(entry)

    return predictions[0]
