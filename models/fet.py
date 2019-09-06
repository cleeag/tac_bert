import numpy as np
from utils import model_utils
import os

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

class fet_model(nn.Module):
    def __init__(self, config, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding ):
        super().__init__()

        self.dropout = config.dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.use_mlp = config.use_mlp
        self.use_bert = config.use_bert
        self.device = device
        self.max_seq_length = config.max_seq_length
        self.mlp_hidden_dim = config.mlp_hidden_dim
        self.bert_hdim = config.bert_hdim
        self.batch_size = config.batch_size
        self.bert_use_four = config.bert_use_four

        # initialize type embedding
        self.type_vocab, self.type_id_dict = type_vocab, type_id_dict
        self.l1_type_indices, self.l1_type_vec, self.child_type_vecs = model_utils.build_hierarchy_vecs(
            self.type_vocab, self.type_id_dict)
        self.n_types = len(self.type_vocab)
        self.type_embed_dim = config.type_embed_dim
        self.type_embeddings = torch.tensor(
            np.random.normal(scale=0.01, size=(self.type_embed_dim, self.n_types)).astype(np.float32),
            device=self.device, requires_grad=True)
        self.type_embeddings = nn.Parameter(self.type_embeddings)

        self.word_vec_dim = embedding_layer.embedding_dim
        self.embedding_layer = embedding_layer


        # build BERT
        if self.use_bert:
            self.bert_model = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-cased',
                                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format('-1')))

        # build DNN
        if not self.bert_use_four:
            linear_map_input_dim = self.bert_hdim + self.word_vec_dim
        else:
            linear_map_input_dim = self.bert_hdim * 4 + self.word_vec_dim
        if not self.use_mlp:
            self.linear_map = nn.Linear(linear_map_input_dim, self.type_embed_dim, bias=False)
            # self.linear_map = nn.Linear(linear_map_input_dim, self.n_types, bias=False)
        else:
            mlp_hidden_dim = linear_map_input_dim // 2 if self.mlp_hidden_dim is None else self.mlp_hidden_dim
            # self.linear_map1 = nn.Linear(linear_map_input_dim, mlp_hidden_dim)
            self.linear_map1 = nn.Linear(linear_map_input_dim, linear_map_input_dim // 2)
            # self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(linear_map_input_dim // 2)
            # self.linear_map2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.linear_map2 = nn.Linear(linear_map_input_dim // 2, mlp_hidden_dim)
            self.lin2_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map3 = nn.Linear(mlp_hidden_dim, self.type_embed_dim)

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs):

        bert_context_hidden, pooled_output = self.bert_model(context_token_seqs, output_all_encoded_layers=True)
        if not self.bert_use_four:
            bert_context_hidden = bert_context_hidden[-1]
        else:
            bert_context_hidden = bert_context_hidden[-4:]
            bert_context_hidden = torch.cat(bert_context_hidden, dim=2)

        context_hidden = bert_context_hidden[list(range(context_token_seqs.size(0))), mention_token_idxs, :]

        name_output = model_utils.get_avg_token_vecs(self.device, self.embedding_layer, mstr_token_seqs)

        cat_output = torch.cat((context_hidden, name_output), dim=1)

        if not self.use_mlp:
            mention_reps = self.linear_map(self.dropout_layer(cat_output))
        else:
            l1_output = self.linear_map1(self.dropout_layer(cat_output))
            # l1_output = F.relu(l1_output)
            l1_output = self.lin1_bn(F.relu(l1_output))
            # mention_reps = self.linear_map2(F.dropout(l1_output, self.dropout, training))
            l2_output = self.linear_map2(self.dropout_layer(l1_output))
            l2_output = self.lin2_bn(F.relu(l2_output))
            mention_reps = self.linear_map3(self.dropout_layer(l2_output))

        logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
                              self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
        logits = logits.view(-1, self.n_types)
        return logits

    def get_loss(self, true_type_vecs, scores, margin=1.0, person_loss_vec=None):
        tmp1 = torch.sum(true_type_vecs * F.relu(margin - scores), dim=1)
        tmp2 = (1 - true_type_vecs) * F.relu(margin + scores)
        if person_loss_vec is not None:
            tmp2 *= person_loss_vec.view(-1, self.n_types)
        tmp2 = torch.sum(tmp2, dim=1)
        loss = torch.mean(torch.add(tmp1, tmp2))
        return loss

    def inference_full(self, logits, extra_label_thres=0.5, is_torch_tensor=True):
        """
        first find the parent type labels, then find children labels according to parent labels.

        :param logits: (batch_size, label_size), prediction scores
        :param extra_label_thres: threshold for extra labels
        :param is_torch_tensor: if logits is a torch tensor
        :return: lsit of list of prediction labels. first is with the highest probability
        """
        if is_torch_tensor:
            logits = logits.data.cpu().numpy()
        l1_type_scores = logits[:, self.l1_type_indices]
        tmp_indices = np.argmax(l1_type_scores, axis=1)
        max_l1_indices = self.l1_type_indices[tmp_indices]
        l2_scores = self.child_type_vecs[max_l1_indices] * logits
        max_l2_indices = np.argmax(l2_scores, axis=1)
        label_preds_main = list()
        for i, (l1_idx, l2_idx) in enumerate(zip(max_l1_indices, max_l2_indices)):
            label_preds_main.append([l2_idx] if l2_scores[i][l2_idx] > 1e-4 else [l1_idx])

        label_preds = list()
        for i in range(len(logits)):
            extra_idxs = np.argwhere(logits[i] > extra_label_thres).squeeze(axis=1)
            label_preds.append(list(set(label_preds_main[i] + list(extra_idxs))))

        return label_preds
