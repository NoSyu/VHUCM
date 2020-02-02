import pickle
import codecs


def load_pickle(path):
    with codecs.open(path, 'rb') as input_f:
        return pickle.load(input_f)
