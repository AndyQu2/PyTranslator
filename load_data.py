import codecs
import random
import numpy as np
import regex

MIN_CUT = 0
MAX_LEN = 50
SOURCE_TRAIN_CN = 'data/train/cn.txt'
SOURCE_TRAIN_EN = 'data/train/en.txt'
SOURCE_TEST_CN = 'data/test/cn.txt'
SOURCE_TEST_EN = 'data/test/en.txt'

def load_vocab(language):
    assert language in ['cn', 'en']
    vocab = [
        line.split()[0] for line in codecs.open(
            'data/preprocessed/{}.txt.vocab.tsv'.format(language), 'r',
            'utf-8').read().splitlines() if int(line.split()[1]) >= MIN_CUT
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_cn_vocab():
    word2idx, idx2word = load_vocab('cn')
    return word2idx, idx2word

def load_en_vocab():
    word2idx, idx2word = load_vocab('en')
    return word2idx, idx2word

def create_data(source_sets, target_sets):
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    x_list, y_list, sources, targets = [], [], [], []
    for source_sent, target_sent in zip(source_sets, target_sets):
        x = [
            cn2idx.get(word, 1)
            for word in ('<S> ' + source_sent + ' </S>').split()
        ]
        y = [
            en2idx.get(word, 1)
            for word in ('<S> ' + target_sent + ' </S>').split()
        ]
        if max(len(x), len(y)) <= MAX_LEN:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            sources.append(source_sent)
            targets.append(target_sent)

    X = np.zeros([len(x_list), MAX_LEN], np.int32)
    Y = np.zeros([len(y_list), MAX_LEN], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.pad(x, [0, MAX_LEN - len(x)],
                          'constant',
                          constant_values=(0, 0))
        Y[i] = np.pad(y, [0, MAX_LEN - len(y)],
                          'constant',
                          constant_values=(0, 0))

    return X, Y, sources, targets


def load_data(data_type):
    if data_type == 'train':
        source, target = SOURCE_TRAIN_CN, SOURCE_TRAIN_EN
    elif data_type == 'test':
        source, target = SOURCE_TEST_CN, SOURCE_TEST_EN
    assert data_type in ['train', 'test']
    cn_sets = [
        regex.sub("[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(source, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]
    en_sets = [
        regex.sub("[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(target, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]

    x, y, sources, targets = create_data(cn_sets, en_sets)
    return x, y, sources, targets


def load_train_data():
    x, y, _, _ = load_data('train')
    return x, y


def load_test_data():
    x, y, _, _ = load_data('test')
    return x, y


def get_batch_indices(total_length, batch_size):
    assert (batch_size <=
            total_length), ('Batch size is large than total data length.'
                            'Check your data or change batch size.')
    current_index = 0
    indexes = [i for i in range(total_length)]
    random.shuffle(indexes)
    while True:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexes[current_index:current_index + batch_size], current_index

def idx_to_sentence(arr, vocab, insert_space=False):
    res = ''
    first_word = True
    for id in arr:
        word = vocab[id.item()]
        if insert_space and not first_word:
            res += ' '
        first_word = False
        res += word
    return res