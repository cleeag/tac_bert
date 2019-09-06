import pandas as pd
import pickle as pkl
from os.path import join
from collections import defaultdict
from tqdm import tqdm
import json

data_dir = '/data/cleeag/tac19'
res_dir = 'res'







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

        pkl.dump(pageID2type_dict, file=open(join(data_dir, 'pageID2type.pkl'),'wb'))
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

    pkl.dump(mention2type_dict, file=open(join(data_dir, 'mention2type_dict.pkl'),'wb'))


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
    type_count_ls.sort(key=lambda x:x[1], reverse=True)

    with open(join(data_dir, 'type_count_dict.txt'), 'w') as w:
        for pair in type_count_ls:
            pair = [str(_) for _ in pair]
            write_pair = '\t'.join(pair) + '\n'
            w.write(write_pair)


def get_top_focus_types():
    type_count_dict = ''


if __name__ == '__main__':
    # test = True
    test = False
    get_type_mention_count_dict(test=test)
    get_type_count(test=test)
    # check_yago_types_of_at_least_10()
