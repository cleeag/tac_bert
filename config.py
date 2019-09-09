from platform import platform
from os.path import join
import socket

if platform().startswith('Windows'):
    PLATFORM = 'Windows'
    DATA_DIR = 'd:/data/fet'
else:
    PLATFORM = 'Linux'
    DATA_DIR = '/data/cleeag/fetel'

TOKEN_UNK = '<UNK>'
TOKEN_ZERO_PAD = '<ZPAD>'
TOKEN_EMPTY_PAD = '<EPAD>'
TOKEN_MENTION = '<MEN>'

RANDOM_SEED = 771
NP_RANDOM_SEED = 7711
PY_RANDOM_SEED = 9973

MACHINE_NAME = socket.gethostname()
RES_DIR = join(DATA_DIR, 'res')
EL_DATA_DIR = join(DATA_DIR, 'el')
MODEL_DIR = join(DATA_DIR, 'models')
LOG_DIR = join(DATA_DIR, 'log')

EL_CANDIDATES_DATA_FILE = join(RES_DIR, 'enwiki-20151002-candidate-gen.pkl')
WIKI_FETEL_WORDVEC_FILE = join(RES_DIR, 'enwiki-20151002-nef-wv-glv840B300d.pkl')
WIKI_ANCHOR_SENTS_FILE = join(RES_DIR, 'enwiki-20151002-anchor-sents.txt')

FIGER_FILES = {
    'typed-wiki-mentions': join(DATA_DIR, 'Wiki/enwiki-20151002-anchor-mentions-typed.txt'),
    'anchor-train-data-prefix': join(DATA_DIR, 'Wiki/enwiki20151002anchor-fetwiki-0_1'),
    'anchor-train-data-prefix-bert': join(DATA_DIR, 'Wiki/enwiki20151002anchor-fetwiki-0_1-bert'),
    'type-vocab': join(DATA_DIR, 'Wiki/figer-type-vocab.txt'),
    'wid-type-file': join(DATA_DIR, 'Wiki/wid-types-figer.txt'),
    'fetel-test-mentions': join(DATA_DIR, 'Wiki/figer-fetel-test-mentions.json'),
    'fetel-test-sents': join(DATA_DIR, 'Wiki/figer-fetel-test-sents.json'),
}

BBN_FILES = {
}


max_seq_length = 128
# eval_batch_size = 50
dropout = 0.5

# dimensions
lstm_hidden_dim = 150
type_embed_dim = 500
pred_mlp_hdim = 400
mlp_hidden_dim = 400
bert_hdim = 768

# training parameters
n_iter = 15
bert_adam_warmup = 0.4
lr_gamma = 0.7
nil_rate = 0.5
rand_per = True
per_penalty = 2.0

# model structure
use_gpu = 1
use_bert = 1
bert_use_four = True
use_lstm = 0
concat_lstm = False
use_mlp = 0
# test = True
test = False

batch_size = 46 if use_bert else 256
# learning_rate = 0.001
learning_rate = 3e-5 if use_bert else 0.001