import os
import argparse
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
from layers.rnncells import StackedGRUCell

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = nn.GRU
                if key == 'rnncell':
                    value = StackedGRUCell
                setattr(self, key, value)

        project_dir = Path(__file__).resolve().parent.parent
        data_dir = project_dir.joinpath('datasets')
        save_dir = project_dir.joinpath('results')

        self.dataset_dir = data_dir.joinpath(self.data_name)

        self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')

        self.convs_path = self.data_dir.joinpath('convs.pkl')
        self.utterances_length_path = self.data_dir.joinpath('utterances_length.pkl')
        self.conversations_length_path = self.data_dir.joinpath('conversations_length.pkl')
        self.convs_users_path = self.data_dir.joinpath('convs_users.pkl')

        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_path = save_dir.joinpath(self.data_name, self.model, time_now)
            self.logdir = str(self.save_path)
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = str(self.save_path)

        if self.pretrained_wv:
            self.pretrained_wv_path = self.dataset_dir.joinpath("fasttext_wv.pkl")
            self.embedding_size = 300
        else:
            self.pretrained_wv_path = None

        if self.pretrained_uv:
            self.pretrained_uv_path = self.dataset_dir.joinpath("user_edge.pkl")
        else:
            self.pretrained_uv_path = None

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--eval_batch_size', type=int, default=80)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--max_unroll', type=int, default=30)
    parser.add_argument('--sample', type=str2bool, default=False,
                        help='if false, use beam search for decoding')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1)

    parser.add_argument('--model', type=str, default='HRED')
    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--rnncell', type=str, default='gru')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--tie_embedding', type=str2bool, default=True)
    parser.add_argument('--encoder_hidden_size', type=int, default=500)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--decoder_hidden_size', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--context_size', type=int, default=500)
    parser.add_argument('--feedforward', type=str, default='FeedForward')
    parser.add_argument('--activation', type=str, default='Tanh')
    parser.add_argument('--users', type=str2bool, default=False)

    parser.add_argument('--z_utter_size', type=int, default=100)
    parser.add_argument('--z_conv_size', type=int, default=100)
    parser.add_argument('--word_drop', type=float, default=0.0,
                        help='only applied to variational models')
    parser.add_argument('--kl_threshold', type=float, default=0.0)
    parser.add_argument('--kl_annealing_iter', type=int, default=5000)
    parser.add_argument('--importance_sample', type=int, default=100)
    parser.add_argument('--sentence_drop', type=float, default=0.25)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--n_context', type=int, default=1)
    parser.add_argument('--n_sample_step', type=int, default=1)

    parser.add_argument('--bow', type=str2bool, default=False)

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=10)

    parser.add_argument('--data_name', type=str, default='tc_10_15')
    parser.add_argument('--pretrained_wv', type=str2bool, default=True)
    parser.add_argument('--pretrained_uv', type=str2bool, default=False)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
