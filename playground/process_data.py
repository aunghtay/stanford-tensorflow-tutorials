from collections import Counter
import numpy as np
import os
import random
from six.moves import urllib
import tensorflow as tf
import zipfile

FILE_NAME = 'text8.zip'
EXPECTED_BYTES = 31344016
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
DATA_FOLDER = 'data/'


def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded. """
    os.makedirs(DATA_FOLDER, exist_ok=True)

    file_path = DATA_FOLDER + file_name
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path

    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file ', file_name)
    else:
        raise Exception('File ' + file_name + ' might be corrupted.')

    return file_path


def read_data(file_path):
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words


def build_vocab(words, vocab_size):
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    os.makedirs('processed', exist_ok=True)
    with open('processed/vocab_1000.tsv', 'w') as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + '\n')
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary


def convert_words_to_index(words, dictionary):
    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_sample(index_words, context_window_size):
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        for target in index_words[max(0, index - context): index]:
            yield center, target
        for target in index_words[index + 1: index + context + 1]:
            yield center, target


def get_batch(iterator, batch_size):
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch


def process_data(vocab_size, batch_size, skip_window):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)
