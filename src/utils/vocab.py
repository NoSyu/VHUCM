from collections import defaultdict
import pickle
import torch
from torch import Tensor
from torch.autograd import Variable
from nltk import FreqDist
from .convert import to_tensor, to_var

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 1, 2, 3]


class Vocab(object):
    def __init__(self, tokenizer=None, max_size=None, min_freq=1):
        self.vocab_size = 0
        self.freqdist = FreqDist()
        self.tokenizer = tokenizer

    def update(self, max_size=None, min_freq=1):
        self.id2word = {
            PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN,
            SOS_ID: SOS_TOKEN, EOS_ID: EOS_TOKEN
        }
        self.word2id = defaultdict(lambda: UNK_ID)  # Not in vocab => return UNK
        self.word2id.update({
            PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
            SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID
        })

        vocab_size = 4
        min_freq = max(min_freq, 1)

        freqdist = self.freqdist.copy()
        special_freqdist = {token: freqdist[token]
                            for token in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]}
        freqdist.subtract(special_freqdist)

        sorted_frequency_counter = sorted(freqdist.items(), key=lambda k_v: k_v[0])
        sorted_frequency_counter.sort(key=lambda k_v: k_v[1], reverse=True)

        for word, freq in sorted_frequency_counter:
            if freq < min_freq or vocab_size == max_size:
                break
            self.id2word[vocab_size] = word
            self.word2id[word] = vocab_size
            vocab_size += 1

        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.id2word)

    def load(self, word2id_path=None, id2word_path=None):
        if word2id_path:
            with open(word2id_path, 'rb') as f:
                word2id = pickle.load(f)
            self.word2id = defaultdict(lambda: UNK_ID)
            self.word2id.update(word2id)
            self.vocab_size = len(self.word2id)

        if id2word_path:
            with open(id2word_path, 'rb') as f:
                id2word = pickle.load(f)
            self.id2word = id2word

    def add_word(self, word):
        assert isinstance(word, str), 'Input should be str'
        self.freqdist.update([word])

    def add_sentence(self, sentence, tokenized=False):
        if not tokenized:
            sentence = self.tokenizer(sentence)
        for word in sentence:
            self.add_word(word)

    def add_dataframe(self, conversation_df, tokenized=True):
        for conversation in conversation_df:
            for sentence in conversation:
                self.add_sentence(sentence, tokenized=tokenized)

    def pickle(self, word2id_path, id2word_path):
        with open(word2id_path, 'wb') as f:
            pickle.dump(dict(self.word2id), f)

        with open(id2word_path, 'wb') as f:
            pickle.dump(self.id2word, f)

    def to_list(self, list_like):
        if isinstance(list_like, list):
            return list_like

        if isinstance(list_like, Variable):
            return list(to_tensor(list_like).numpy())
        elif isinstance(list_like, Tensor):
            return list(list_like.numpy())

    def id2sent(self, id_list):
        id_list = self.to_list(id_list)
        sentence = []
        for id in id_list:
            word = self.id2word[id]
            if word not in [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN]:
                sentence.append(word)
            if word == EOS_TOKEN:
                break
        return sentence

    def sent2id(self, sentence, var=False):
        id_list = [self.word2id[word] for word in sentence]
        if var:
            id_list = to_var(torch.LongTensor(id_list), eval=True)
        return id_list

    def decode(self, id_list):
        sentence = self.id2sent(id_list)
        return ' '.join(sentence)
