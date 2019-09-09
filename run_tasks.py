from utils import exp_utils, datautils, utils
import pickle
import config
from tqdm import tqdm

if __name__ == "__main__":
    task = input('input task: ')
    if task == 'prep':
        # job = 'train'
        job = 'dev'
        # data_pkl = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-bert-{job}.pkl"
        data_pkl = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-{job}.pkl"
        print('loading training data {} ...'.format(data_pkl), end=' ', flush=True)
        samples = datautils.load_pickle_data(data_pkl)
        print('done', flush=True)
        out_list = []
        for sample in tqdm(samples):
            # new_s = [sample[0], sample[6][sample[2]:sample[3]], sample[7], sample[8], sample[5]]
            new_s = (sample[0], sample[6][sample[2]:sample[3]], sample[6], sample[2], sample[5])
            out_list.append(new_s)

        # output_file = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-bert-{job}-slim.pkl"
        output_file = f"/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-{job}-slim.pkl"
        pickle.dump(out_list, file=open(output_file, 'wb'))

    elif task == 'r':
        train_data_pkl = "/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-bert-dev.pkl"
        samples = datautils.load_pickle_data(train_data_pkl)
        print(samples[0])

    elif task == 'ib':
        from collections import defaultdict

        train_data_pkl = "/data/cleeag/fetel/Wiki/enwiki20151002anchor-fetwiki-0_1-train-slim.pkl"
        samples = datautils.load_pickle_data(train_data_pkl)
        seq_len_dict = defaultdict(int)
        for s in tqdm(samples):
            seq_len_dict[len(s[2])] += 1
        seq_len_ls = [(k, v) for k, v in seq_len_dict.items()]
        seq_len_ls.sort(key=lambda x:x[0], reverse=True)
        long_count, total_count = 0, 0
        for x in seq_len_ls:
            if x[0] > 128:
                long_count += x[1]
            total_count += x[1]
            print(x)
        print(long_count, total_count, long_count/total_count)

    else:
        print('None!')