import torch
from torch import nn
from tqdm import tqdm
import logging
from collections import namedtuple
import random

import config
from utils import utils, datautils

ModelSample = namedtuple('ModelSample', [
    'mention_id',
    'mstr_token_seq',
    'context_token_seq_bert',
    'mention_token_idx_bert',
    'labels'
])


# class ModelSample:
#     def __init__(self,
#                  mention_id,
#                  mstr_token_seq,
#                  context_token_seq_bert,
#                  mention_token_idx_bert,
#                  full_labels):
#         self.mention_id = mention_id
#         self.mstr_token_seq = mstr_token_seq
#         self.context_token_seq_bert = context_token_seq_bert
#         self.mention_token_idx_bert = mention_token_idx_bert
#         self.full_labels = full_labels


class GlobalRes:
    def __init__(self, type_vocab_file, word_vecs_file):
        self.type_vocab, self.type_id_dict = datautils.load_type_vocab(type_vocab_file)
        self.parent_type_ids_dict = utils.get_parent_type_ids_dict(self.type_id_dict)
        self.n_types = len(self.type_vocab)

        print('loading {} ...'.format(word_vecs_file), end=' ', flush=True)
        self.token_vocab, self.token_vecs = datautils.load_pickle_data(word_vecs_file)
        self.token_id_dict = {t: i for i, t in enumerate(self.token_vocab)}
        print('done', flush=True)
        self.zero_pad_token_id = self.token_id_dict[config.TOKEN_ZERO_PAD]
        self.mention_token_id = self.token_id_dict[config.TOKEN_MENTION]
        self.unknown_token_id = self.token_id_dict[config.TOKEN_UNK]
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(self.token_vecs))
        self.embedding_layer.padding_idx = self.zero_pad_token_id
        self.embedding_layer.weight.requires_grad = False
        self.embedding_layer.share_memory()


def anchor_samples_to_model_samples_bert(config, samples, parent_type_ids_dict):
    model_samples = list()
    skip_count = 0
    for i, sample in enumerate(tqdm(samples)):
        if sample[8] >= config.max_seq_length:
            skip_count += 1
            continue
        mention_id = sample[0]
        # mention_str = sample[1]
        pos_beg, pos_end = sample[2], sample[3]
        mstr_token_seq = sample[6][pos_beg:pos_end]
        # mention_token_idx = pos_beg
        # context_token_seq = sample[6][:pos_beg] + [mention_token_id] + sample[6][pos_end:]
        context_token_seq_bert = sample[7]
        while len(context_token_seq_bert) < config.max_seq_length:
            context_token_seq_bert.append(0)
        mention_token_idx_bert = sample[8]

        # if len(context_token_seq) > 256:
        #     context_token_seq = context_token_seq[:256]
        # if mention_token_idx >= 256:
        #     mention_token_idx = 255

        full_labels = utils.get_full_type_ids(sample[5], parent_type_ids_dict)

        model_samples.append(ModelSample(mention_id,
                                         mstr_token_seq,
                                         context_token_seq_bert,
                                         mention_token_idx_bert,
                                         full_labels))
        # model_samples.append([mention_id,
        #                       mstr_token_seq,
        #                       context_token_seq_bert,
        #                       mention_token_idx_bert,
        #                       full_labels])
    # logging.info(f'skipped {skip_count} entries')
    return model_samples


def samples_to_tensor(config, device, gres, samples, person_type_id=None, l2_person_type_ids=None, rand=True):
    """
    format of "samples":
    0: mention_id
    1: mstr_token_seqs, list of int : token of mention, using hldai's tokenization method
    2: context_token, list of int : token of context, used [MASK] to substitute the mention token.
        tokenized with bert's tokenization method
    3: mention_token_idx, list of int : index of the position of the mention token, used in fet_bert model
        to retrieve context representation
    4: labels, list of int :


    """
    mstr_token_seqs = [s[1] for s in samples]

    context_token_tensor = torch.zeros(len(samples), config.max_seq_length, device=device, dtype=torch.long)
    context_token_list = [torch.tensor(s[2][:config.max_seq_length]) for s in samples]
    context_token_list = torch.nn.utils.rnn.pad_sequence(context_token_list, batch_first=True, padding_value=0)
    context_token_tensor[:, :min(context_token_list.size(1), config.max_seq_length)] = context_token_list

    mention_token_idx_tensor = torch.tensor([s[3] if s[3] < config.max_seq_length else config.max_seq_length - 1
                                             for s in samples], device=device, dtype=torch.long)

    if not rand:
        type_vecs = torch.tensor([
            utils.onehot_encode(
            utils.get_full_type_ids(s[4], gres.parent_type_ids_dict), gres.n_types) for s in samples],
            dtype=torch.float32, device=device)
    else:
        type_vecs = list()
        for sample in samples:
            labels = utils.get_full_type_ids(sample[4], gres.parent_type_ids_dict)
            type_vec = utils.onehot_encode(labels, gres.n_types)
            if person_type_id is not None and person_type_id in labels:
                for _ in range(3):
                    rand_person_type_id = l2_person_type_ids[random.randint(0, len(l2_person_type_ids) - 1)]
                    if type_vec[rand_person_type_id] < 1.0:
                        type_vec[rand_person_type_id] = 1.0
                        break
            type_vecs.append(type_vec)
        type_vecs = torch.tensor(type_vecs, dtype=torch.float32, device=device)

    return context_token_tensor, mention_token_idx_tensor, mstr_token_seqs, type_vecs


def get_mstr_context_batch_input_rand_per_bert(device, n_types, samples, person_type_id=None,
                                               person_l2_type_ids=None, max_seq_length=256, rand=True):
    # make context_token_seqs_list into torch tensor
    context_token_seqs_list = [torch.tensor(s.context_token_seq_bert, device=device) for s in samples]
    context_token_seqs = torch.zeros(len(samples), max_seq_length, device=device)
    for i in range(len(context_token_seqs_list)):
        context_token_seqs[i][:len(context_token_seqs_list[i])] \
            = context_token_seqs_list[i][:max_seq_length]

    mention_token_idxs = torch.tensor([s.mention_token_idx_bert for s in samples], device=device)
    mstrs = [s.mention_str for s in samples]

    # make mstr_token_seqs into torch tensor
    mstr_token_seqs_list = [torch.tensor(s.mstr_token_seq, device=device) for s in samples]
    mstr_token_seqs = torch.zeros(len(samples), 32, device=device)
    for i in range(len(mstr_token_seqs_list)):
        mstr_token_seqs[i][:len(mstr_token_seqs_list[i])] = mstr_token_seqs_list[i]

    if not rand:
        type_vecs = torch.tensor([utils.onehot_encode(s.labels, n_types) for s in samples],
                                 dtype=torch.float32, device=device)
    else:
        type_vecs = list()
        for sample in samples:
            type_vec = utils.onehot_encode(sample.labels, n_types)
            if person_type_id is not None and person_type_id in sample.labels:
                for _ in range(3):
                    rand_person_type_id = person_l2_type_ids[random.randint(0, len(person_l2_type_ids) - 1)]
                    if type_vec[rand_person_type_id] < 1.0:
                        type_vec[rand_person_type_id] = 1.0
                        break
            type_vecs.append(type_vec)
        type_vecs = torch.tensor(type_vecs, dtype=torch.float32, device=device)
    return context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs, type_vecs


def get_l2_person_type_ids(type_vocab):
    person_type_ids = list()
    for i, t in enumerate(type_vocab):
        if t.startswith('/person') and t != '/person':
            person_type_ids.append(i)
    return person_type_ids
