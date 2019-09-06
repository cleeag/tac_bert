import datetime
import torch
import numpy as np
import os
import logging
from utils.loggingutils import init_universal_logging
import random

import torch

from utils import exp_utils, datautils, model_utils
from models.fet import fet_model
import config


def train_model(test=False):
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    # device = torch.device('cpu')
    device_name = torch.cuda.get_device_name(device) if device is not 'cpu' else 'cpu'
    logging.info(f'running on device: {device_name}')
    dataset = 'figer'
    datafiles = config.FIGER_FILES
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE
    save_model_file = config.DATA_DIR + 'models' + 'test'

    data_prefix = datafiles['anchor-train-data-prefix-bert']
    # dev_data_pkl = data_prefix + '-dev.pkl'
    # train_data_pkl = data_prefix + '-train.pkl'
    dev_data_pkl = data_prefix + '-dev-slim.pkl'
    if test:
        train_data_pkl = data_prefix + '-dev-slim.pkl'
    else:
        train_data_pkl = data_prefix + '-train-slim.pkl'
    dev_results_file = os.path.join(config.DATA_DIR, 'Wiki/fetel-deep-results-{}.txt'.format(dataset))

    gres = exp_utils.GlobalRes(datafiles['type-vocab'], word_vecs_file)
    logging.info('dataset={}'.format(dataset))

    logging.info(
        'type_embed_dim={} cxt_lstm_hidden_dim={} pmlp_hdim={} amlp_hdim={}'.format(
            config.type_embed_dim, config.context_lstm_hidden_dim, config.pred_mlp_hdim,
            config.att_mlp_hdim))
    logging.info('rand_per={} per_pen={}'.format(config.rand_per, config.per_penalty))

    print('loading training data {} ...'.format(train_data_pkl), end=' ', flush=True)
    training_samples = datautils.load_pickle_data(train_data_pkl)
    print('done', flush=True)
    # training_samples = exp_utils.anchor_samples_to_model_samples_bert(config, samples, gres.parent_type_ids_dict)

    print('loading dev data {} ...'.format(dev_data_pkl), end=' ', flush=True)
    dev_samples = datautils.load_pickle_data(dev_data_pkl)
    print('done', flush=True)
    # dev_samples = exp_utils.anchor_samples_to_model_samples_bert(config, dev_data, gres.parent_type_ids_dict)
    dev_true_labels_dict = {s[0]: [gres.type_vocab[l] for l in s[4]] for s in dev_samples}

    logging.info('building model...')
    model = fet_model(config, device, gres.type_vocab, gres.type_id_dict, gres.embedding_layer)
    model.to(device)

    logging.info('{}'.format(model.__class__.__name__))
    logging.info('training batch size: {}'.format(config.batch_size))

    # get person penalty vector
    person_type_id = gres.type_id_dict.get('/person')
    l2_person_type_ids = None
    person_loss_vec = None
    if person_type_id is not None:
        l2_person_type_ids = exp_utils.get_l2_person_type_ids(gres.type_vocab)
        person_loss_vec = np.ones(gres.n_types, np.float32)
        for tid in l2_person_type_ids:
            person_loss_vec[tid] = config.per_penalty
        person_loss_vec = torch.tensor(person_loss_vec, dtype=torch.float32, device=device)

    n_batches = (len(training_samples) + config.batch_size - 1) // config.batch_size
    n_steps = config.n_iter * n_batches
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_batches, gamma=config.lr_gamma)
    losses = list()
    best_dev_acc = -1

    # start training
    logging.info('{} steps, {} steps per iter, lr_decay={}, start training ...'.format(
        config.n_iter * n_batches, n_batches, config.lr_gamma))
    step = 0
    while True:
        if step == n_steps:
            break

        batch_idx = step % n_batches
        batch_beg, batch_end = batch_idx * config.batch_size, min((batch_idx + 1) * config.batch_size,
                                                                  len(training_samples))
        context_token_seqs, mention_token_idxs, mstr_token_seqs, type_vecs \
            = exp_utils.samples_to_tensor(
            config, device, gres, training_samples[batch_beg:batch_end],
            person_type_id, l2_person_type_ids)
        model.train()
        logits = model(context_token_seqs, mention_token_idxs, mstr_token_seqs)
        loss = model.get_loss(type_vecs, logits, person_loss_vec=person_loss_vec)
        scheduler.step()
        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, float('inf'))
        optimizer.step()
        losses.append(loss.data.cpu().numpy())
        logging.info('step={}/{} accumulated loss = {:.4f}, loss = {:.4f}'.format(step, n_steps, sum(losses), loss))

        step += 1

        eval_cycle = 3 if config.test else 100
        if step % eval_cycle == 0:
            l_v, acc_v, pacc_v,  maf1, mif1, dev_results = \
                model_utils.eval_fetel(config, gres, model, dev_samples, dev_true_labels_dict)
            best_tag = '*' if acc_v > best_dev_acc else ''
            # logging.info(
            #     'step={}/{} l={:.4f} l_v={:.4f} acc_v={:.4f} paccv={:.4f}{}\n'.format(
            #         step, n_steps, loss, l_v, acc_v, pacc_v, best_tag))
            print('   evaluation result: ')
            print('     l_v={:.4f} acc_v={:.4f} paccv={:.4f} macro_f1={:.4f} micro_f1={:.4f}{}\n'.format(
                    l_v, acc_v, pacc_v, maf1, mif1, best_tag))
            if acc_v > best_dev_acc and save_model_file:
                # torch.save(model.state_dict(), save_model_file)
                logging.info('model saved to {}'.format(save_model_file))

            if dev_results_file is not None and acc_v > best_dev_acc:
                # datautils.save_json_objs(dev_results, dev_results_file)
                logging.info('dev reuslts saved {}'.format(dev_results_file))

            if acc_v > best_dev_acc:
                best_dev_acc = acc_v
        # losses = list()
            if config.test:
                input('proceed? ')


        pass




if __name__ == '__main__':
    torch.random.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.NP_RANDOM_SEED)
    random.seed(config.PY_RANDOM_SEED)
    str_today = datetime.date.today().strftime('%d-%m-%Y')
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], str_today, config.MACHINE_NAME))
    init_universal_logging(log_file, mode='a', to_stdout=True)

    train_model(config.test)
