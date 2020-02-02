import torch
import torch.nn as nn
import models
from utils import TensorboardWriter
import os
import re


class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model
        self.writer = None
        self.optimizer = None
        self.epoch_loss = None
        self.validation_loss = None

    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)

            if self.config.mode == 'train' and self.config.checkpoint is None:
                print('Parameter initiailization')
                for name, param in self.model.named_parameters():
                    if 'weight_hh' in name:
                        print('\t' + name)
                        nn.init.orthogonal_(param)

                    if 'bias_hh' in name:
                        print('\t' + name)
                        dim = int(param.size(0) / 3)
                        param.data[dim:2 * dim].fill_(2.0)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            self.writer = TensorboardWriter(self.config.logdir)
            self.optimizer = self.config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                   lr=self.config.learning_rate)

    def save_model(self, epoch):
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        self.model.load_state_dict(torch.load(checkpoint))

    def write_summary(self, epoch_i):
        epoch_loss = getattr(self, 'epoch_loss', None)
        if epoch_loss is not None:
            self.writer.update_loss(loss=epoch_loss, step_i=epoch_i + 1, name='train_loss')

        epoch_recon_loss = getattr(self, 'epoch_recon_loss', None)
        if epoch_recon_loss is not None:
            self.writer.update_loss(loss=epoch_recon_loss, step_i=epoch_i + 1, name='train_recon_loss')

        epoch_kl_div = getattr(self, 'epoch_kl_div', None)
        if epoch_kl_div is not None:
            self.writer.update_loss(loss=epoch_kl_div, step_i=epoch_i + 1, name='train_kl_div')

        kl_mult = getattr(self, 'kl_mult', None)
        if kl_mult is not None:
            self.writer.update_loss(loss=kl_mult, step_i=epoch_i + 1, name='kl_mult')

        epoch_bow_loss = getattr(self, 'epoch_bow_loss', None)
        if epoch_bow_loss is not None:
            self.writer.update_loss(loss=epoch_bow_loss, step_i=epoch_i + 1, name='bow_loss')

        validation_loss = getattr(self, 'validation_loss', None)
        if validation_loss is not None:
            self.writer.update_loss(loss=validation_loss, step_i=epoch_i + 1, name='validation_loss')

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def export_samples(self, beam_size=5):
        raise NotImplementedError
