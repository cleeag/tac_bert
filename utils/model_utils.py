import numpy as np

import torch
from torch import nn

from utils import exp_utils, utils

def build_hierarchy_vecs(type_vocab, type_to_id_dict):
    from utils import utils

    n_types = len(type_vocab)
    l1_type_vec = np.zeros(n_types, np.float32)
    l1_type_indices = list()
    child_type_vecs = np.zeros((n_types, n_types), np.float32)
    for i, t in enumerate(type_vocab):
        p = utils.get_parent_type(t)
        if p is None:
            l1_type_indices.append(i)
            l1_type_vec[type_to_id_dict[t]] = 1
        else:
            child_type_vecs[type_to_id_dict[p]][type_to_id_dict[t]] = 1
    l1_type_indices = np.array(l1_type_indices, np.int32)
    return l1_type_indices, l1_type_vec, child_type_vecs

def get_len_sorted_context_seqs_input(device, context_token_list, mention_token_idxs):
    data_tups = list(enumerate(zip(context_token_list, mention_token_idxs)))
    data_tups.sort(key=lambda x: -len(x[1][0]))
    sorted_seqs = [x[1][0] for x in data_tups]
    mention_token_idxs = [x[1][1] for x in data_tups]
    idxs = [x[0] for x in data_tups]
    back_idxs = [0] * len(idxs)
    for i, idx in enumerate(idxs):
        back_idxs[idx] = i

    back_idxs = torch.tensor(back_idxs, dtype=torch.long, device=device)
    # seqs, seq_lens = get_seqs_torch_input(device, seqs)
    seq_lens = torch.tensor([len(seq) for seq in sorted_seqs], dtype=torch.long, device=device)
    sorted_seqs_tensor_list = [torch.tensor(seq, dtype=torch.long, device=device) for seq in sorted_seqs]
    padded_sorted_seqs_tensor = torch.nn.utils.rnn.pad_sequence(sorted_seqs_tensor_list, batch_first=True)
    mention_token_idxs = torch.tensor(mention_token_idxs, dtype=torch.long, device=device)
    return padded_sorted_seqs_tensor, seq_lens, mention_token_idxs, back_idxs


def get_avg_token_vecs(device, embedding_layer: nn.Embedding, token_seqs):
    lens = torch.tensor([len(seq) for seq in token_seqs], dtype=torch.float32, device=device
                        ).view(-1, 1)
    seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in token_seqs]
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                           padding_value=embedding_layer.padding_idx)
    token_vecs = embedding_layer(seqs)
    vecs_avg = torch.div(torch.sum(token_vecs, dim=1), lens)
    return vecs_avg

def eval_fetel(config, gres: exp_utils.GlobalRes, model, samples, true_labels_dict):
    model.eval()
    batch_size = config.batch_size
    n_batches = (len(samples) + batch_size - 1) // batch_size
    losses = list()
    pred_labels_dict = dict()
    result_objs = list()
    if hasattr(model, 'module'):
        device = model.module.device
    else:
        device = model.device

    for i in range(n_batches):
        batch_beg, batch_end = i * batch_size, min((i + 1) * batch_size, len(samples))
        batch_samples = samples[batch_beg:batch_end]

        context_token_seqs, mention_token_idxs, mstr_token_seqs, type_vecs \
            = exp_utils.samples_to_tensor(
            config, device, gres, batch_samples, rand=False)

        with torch.no_grad():
            logits = model(context_token_seqs, mention_token_idxs, mstr_token_seqs)
            if hasattr(model, 'module'):
                loss = model.module.get_loss(type_vecs, logits)
                preds = model.module.inference_full(logits, extra_label_thres=0.0)

            else:
                loss = model.get_loss(type_vecs, logits)
                preds = model.inference_full(logits, extra_label_thres=0.0)

        losses.append(loss)

        for j, (sample, type_ids_pred, sample_logits) in enumerate(
                zip(batch_samples, preds, logits.data.cpu().numpy())):
            labels = utils.get_full_types([gres.type_vocab[tid] for tid in type_ids_pred])
            pred_labels_dict[sample[0]] = labels
            result_objs.append({'mention_id': sample[0], 'labels': labels,
                                'logits': [float(v) for v in sample_logits]})

    strict_acc = utils.strict_acc(true_labels_dict, pred_labels_dict)
    partial_acc = utils.partial_acc(true_labels_dict, pred_labels_dict)
    maf1 = utils.macrof1(true_labels_dict, pred_labels_dict)
    mif1 = utils.microf1(true_labels_dict, pred_labels_dict)
    return sum(losses), strict_acc, partial_acc, maf1, mif1, result_objs