import pickle as pkl
from os.path import join
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
import sys
sys.path.insert(0, '..')

import pandas as pd
from pytorch_transformers import BertTokenizer
import config

from utils import datautils

data_dir = '/data/cleeag/tac19'
res_dir = 'res'
home_path = '/home/cleeag/tac_bert'


def slim_pickles():
    """
    format of "samples":
    0: mention_id
    1: mstr_token_seqs, list of int : token of mention, using hldai's tokenization method
    2: context_token, list of int : token of context, used [MASK] to substitute the mention token.
        tokenized with bert's tokenization method
    3: mention_token_idx, list of int : index of the position of the mention token, used in fet_bert model
        to retrieve context representation
    4: labels, list of int : not full label, needs to call utils.get_full_types() to obtain full label
    """
    job = 'train'
    # job = 'dev'
    data_pkl = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-bert-{job}.pkl"
    # data_pkl = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-{job}.pkl"
    print('loading training data {} ...'.format(data_pkl), end=' ', flush=True)
    samples = datautils.load_pickle_data(data_pkl)
    print('done', flush=True)
    out_list = []
    for sample in tqdm(samples):
        new_s = [sample[0], sample[6][sample[2]:sample[3]], sample[7], sample[8], sample[5]]
        # new_s = (sample[0], sample[6][sample[2]:sample[3]], sample[6], sample[2], sample[5])
        out_list.append(new_s)

    output_file = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-bert-{job}-slim.pkl"
    # output_file = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-{job}-slim.pkl"
    pkl.dump(out_list, file=open(output_file, 'wb'))


def gen_training_data_from_wiki(typed_mentions_file, sents_file, word_vecs_pkl, sample_rate,
                                n_dev_samples, output_files_name_prefix, core_title_wid_file=None, do_bert=False):
    np.random.seed(config.RANDOM_SEED)
    print('output file destination: {}'.format(output_files_name_prefix))

    if do_bert:
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        # tokenizer.add_special_tokens({'mention': '[MASK]'})

    core_wids = None
    if core_title_wid_file is not None:
        df = datautils.load_csv(core_title_wid_file)
        core_wids = {wid for _, wid in df.itertuples(False, None)}

    print('loading word vec...')
    token_vocab, token_vecs = datautils.load_pickle_data(word_vecs_pkl)
    token_id_dict = {t: i for i, t in enumerate(token_vocab)}
    unknown_token_id = token_id_dict[config.TOKEN_UNK]

    f_mention = open(typed_mentions_file, encoding='utf-8')
    f_sent = open(sents_file, encoding='utf-8')
    all_samples = list()
    cur_sent = json.loads(next(f_sent))
    mention_id = 0
    for i, line in enumerate(f_mention):
        if (i + 1) % 100000 == 0:
            print(i + 1)
        # if i > 40000:
        #     break

        v = np.random.uniform()
        if v > sample_rate:
            continue

        (wid, mention_str, sent_id, pos_beg, pos_end, target_wid, type_ids
         ) = datautils.parse_typed_mention_file_line(line)
        if core_wids is not None and target_wid not in core_wids:
            continue

        mention_str = mention_str.replace('-LRB-', '(').replace('-RRB-', ')')
        while not (cur_sent['wid'] == wid and cur_sent['sent_id'] == sent_id):
            cur_sent = json.loads(next(f_sent))
        sent_tokens = cur_sent['tokens'].split(' ')
        sent_token_ids = [token_id_dict.get(token, unknown_token_id) for token in sent_tokens]

        if not do_bert:
            sample = (mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids)
        else:
            sent_tokens = sent_tokens[:pos_beg] + ['[MASK]'] + sent_tokens[pos_end:]
            full_sent = ' '.join(sent_tokens)
            tokens = ["[CLS]"]
            t = tokenizer.tokenize(full_sent)
            tokens.extend(t)
            mention_token_idx_bert = 0
            for i, x in enumerate(tokens):
                if x == '[MASK]':
                    mention_token_idx_bert = i
                    break
            tokens.append("[SEP]")
            sent_token_bert_ids = tokenizer.convert_tokens_to_ids(tokens)

            sample = (mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids,
                      sent_token_bert_ids, mention_token_idx_bert)

        mention_id += 1
        all_samples.append(sample)
        # print(i, mention_str)
        # print(sent_token_ids)
        # print()
        if (i + 1) % 100000 == 0:
            print(i + 1, mention_str)
            print(sent_token_ids)
            print()
            print(sent_token_bert_ids)

    f_mention.close()
    f_sent.close()

    dev_samples = all_samples[:n_dev_samples]
    train_samples = all_samples[n_dev_samples:]

    print('shuffling ...', end=' ', flush=True)
    rand_perm = np.random.permutation(len(train_samples))
    train_samples_shuffled = list()
    for idx in rand_perm:
        train_samples_shuffled.append(train_samples[idx])
    train_samples = train_samples_shuffled
    print('done')

    dev_mentions, dev_sents = list(), list()
    for i, sample in enumerate(dev_samples):
        if do_bert:
            mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids, \
            sent_token_bert_ids, mention_token_idx_bert = sample
        else:
            mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids = sample
        mention = {'mention_id': mention_id, 'span': [pos_beg, pos_end], 'str': mention_str, 'sent_id': i}
        sent = {'sent_id': i, 'text': ' '.join([token_vocab[token_id] for token_id in sent_token_ids]),
                'afet-senid': 0, 'file_id': 'null'}
        dev_mentions.append(mention)
        dev_sents.append(sent)
    datautils.save_json_objs(dev_mentions, output_files_name_prefix + '-dev-mentions.txt')
    datautils.save_json_objs(dev_sents, output_files_name_prefix + '-dev-sents.txt')
    print('saving pickle data...')
    datautils.save_pickle_data(dev_samples, output_files_name_prefix + '-dev.pkl')
    datautils.save_pickle_data(train_samples, output_files_name_prefix + '-train.pkl')


def get_type_mention_count_dict(test=True):

    yago_file = 'wiki-title-yago-types.txt'
    mention_file = 'enwiki-20190101-anchor-mentions-notok.txt'

    have_pageID2type_dict = False
    if have_pageID2type_dict:
        pageID2type_dict = pkl.load(file=open(join(data_dir, 'pageID2type.pkl'), 'rb'))
    else:
        pageID2type_dict = {}
        # type_set = set()
        title2yago_df = pd.read_csv(join(data_dir, res_dir, yago_file))
        for i in range(len(title2yago_df)):
            pageID2type_dict[int(title2yago_df.iloc[i, 0])] = title2yago_df.iloc[i, 2]

            if i % 100000 == 0:
                print(i, int(title2yago_df.iloc[i, 0]), title2yago_df.iloc[i, 2])

        # with open(join(data_dir, res_dir, yago_file), 'r') as r:
        #     r.readline()
        #     line = r.readline()
        #     i = 0
        #     while line:
        #         items = line.split(',')
        #         pageID2type_dict[int(items[0])] = items[1:]
        #
        #         # types = items[2].split(';')
        #         # type_set.update(types)
        #         if test:
        #             print(i, int(items[0]), items[1])
        #             if i > 3:
        #                 break
        #
        #         i += 1
        #         if i % 100000 == 0:
        #             print(i, int(items[0]), items[1])
        #
        #         line = r.readline()

        pkl.dump(pageID2type_dict, file=open(join(data_dir, 'pageID2type.pkl'), 'wb'))
        print()
        # type_set = list(type_set)
        # type_set.sort()
        # with open('type_set.txt'):

    mention2type_dict = {}
    with open(join(data_dir, res_dir, mention_file), 'r') as r:
        line = r.readline()
        i = 0
        not_in = 0

        while line:
            items = line.split('\t')
            if int(items[-1]) in pageID2type_dict:
                mention2type_dict[items[1]] = pageID2type_dict[int(items[-1])]
            else:
                not_in += 1
            if test:
                print(i, items[1], items[-1])
                if i > 3:
                    break

            i += 1
            if i % 100000 == 0:
                print(i, not_in, items[1], items[-1])

            line = r.readline()

    pkl.dump(mention2type_dict, file=open(join(data_dir, 'mention2type_dict.pkl'), 'wb'))

def get_type_count(test=True):
    mention2type_dict_path = 'mention2type_dict.pkl'
    mention2type_dict = pkl.load(file=open(join(data_dir, mention2type_dict_path), 'rb'))
    type_count_dict = defaultdict(int)
    i = 0
    for mention, type_ls in tqdm(mention2type_dict.items()):
        types = type_ls.strip().split(';')
        for t in types:
            type_count_dict[t] += 1

        if test:
            for k, v in type_count_dict.items():
                print(k, v)
            if i > 3:
                break
        i += 1
    # print(type_count_dict)
    pkl.dump(type_count_dict, file=open(join(data_dir, 'type_count_dict.pkl'), 'wb'))

    type_count_ls = [[k, v] for k, v in type_count_dict.items()]
    type_count_ls.sort(key=lambda x: x[1], reverse=True)

    with open(join(data_dir, 'type_count_dict.txt'), 'w') as w:
        for pair in type_count_ls:
            pair = [str(_) for _ in pair]
            write_pair = '\t'.join(pair) + '\n'
            w.write(write_pair)


if __name__ == '__main__':
    # test = True
    # test = False
    # get_type_mention_count_dict(test=test)
    # get_type_count(test=test)
    # check_yago_types_of_at_least_10()
    # sys.path.append(home_path)
    # gen_training_data_from_wiki(typed_mentions_file=config.FIGER_FILES['typed-wiki-mentions'],
    #                             sents_file=config.WIKI_ANCHOR_SENTS_FILE,
    #                             word_vecs_pkl=config.WIKI_FETEL_WORDVEC_FILE,
    #                             sample_rate=0.1,
    #                             n_dev_samples=2000,
    #                             output_files_name_prefix=config.FIGER_FILES['anchor-train-data-prefix-bert'],
    #                             do_bert=True)

    slim_pickles()