import numpy as np
import torch
from layers import masked_cross_entropy
from utils import to_var
from tqdm import tqdm
import sys
from .hred_solver import SolverHRED


class SolverVHRED(SolverHRED):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SolverVHRED, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)

    def train(self):
        epoch_loss_history = list()
        kl_mult = 0.0
        min_validation_loss = sys.float_info.max
        patience_cnt = self.config.patience

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = list()
            recon_loss_history = []
            kl_div_history = []
            bow_loss_history = []
            self.model.train()
            n_total_words = 0

            for batch_i, (conversations, conversation_length, sentence_length) \
                    in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # conversations: (batch_size) list of conversations
                #   conversation: list of sentences
                #   sentence: list of tokens
                # conversation_length: list of int
                # sentence_length: (batch_size) list of conversation list of sentence_lengths

                target_conversations = [conv[1:] for conv in conversations]

                utterances = [utter for conv in conversations for utter in conv]
                target_utterances = [utter for conv in target_conversations for utter in conv]
                utterance_length = [l for len_list in sentence_length for l in len_list]
                target_utterance_length = [l for len_list in sentence_length for l in len_list[1:]]
                input_conversation_length = [conv_len - 1 for conv_len in conversation_length]

                utterances = to_var(torch.LongTensor(utterances))
                utterance_length = to_var(torch.LongTensor(utterance_length))
                target_utterances = to_var(torch.LongTensor(target_utterances))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))

                self.optimizer.zero_grad()

                utterances_logits, kl_div = self.model(utterances, utterance_length,
                                                       input_conversation_length, target_utterances, decode=False)

                recon_loss, n_words = masked_cross_entropy(utterances_logits, target_utterances, target_utterance_length)

                batch_loss = recon_loss + kl_mult * kl_div
                batch_loss_history.append(batch_loss.item())
                recon_loss_history.append(recon_loss.item())
                kl_div_history.append(kl_div.item())
                n_total_words += n_words.item()

                if batch_i % self.config.print_every == 0:
                    print_str = f'Epoch: {epoch_i + 1}, iter {batch_i}: ' \
                                f'loss = {batch_loss.item() / n_words.item():.3f}, ' \
                                f'recon = {recon_loss.item() / n_words.item():.3f}, ' \
                                f'kl_div = {kl_div.item() / n_words.item():.3f}'
                    tqdm.write(print_str)

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()
                kl_mult = min(kl_mult + 1.0 / self.config.kl_annealing_iter, 1.0)

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)

            epoch_recon_loss = np.sum(recon_loss_history) / n_total_words
            epoch_kl_div = np.sum(kl_div_history) / n_total_words

            self.kl_mult = kl_mult
            self.epoch_loss = epoch_loss
            self.epoch_recon_loss = epoch_recon_loss
            self.epoch_kl_div = epoch_kl_div

            print_str = f'Epoch {epoch_i + 1} loss average: {epoch_loss:.3f}, ' \
                        f'recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
            if bow_loss_history:
                self.epoch_bow_loss = np.sum(bow_loss_history) / n_total_words
                print_str += f', bow_loss = {self.epoch_bow_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

            if min_validation_loss > self.validation_loss:
                min_validation_loss = self.validation_loss
            else:
                patience_cnt -= 1
                self.save_model(epoch_i)

            if patience_cnt < 0:
                print(f'\nEarly stop at {epoch_i}')
                self.save_model(epoch_i)
                return epoch_loss_history

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def evaluate(self):
        self.model.eval()
        batch_loss_history = []
        recon_loss_history = []
        kl_div_history = []
        n_total_words = 0
        for batch_i, (conversations, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths
            target_conversations = [conv[1:] for conv in conversations]

            utterances = [utter for conv in conversations for utter in conv]
            target_utterances = [utter for conv in target_conversations for utter in conv]
            utterance_length = [l for len_list in sentence_length for l in len_list]
            target_utterance_length = [l for len_list in sentence_length for l in len_list[1:]]
            input_conversation_length = [conv_len - 1 for conv_len in conversation_length]

            with torch.no_grad():
                utterances = to_var(torch.LongTensor(utterances))
                utterance_length = to_var(torch.LongTensor(utterance_length))
                target_utterances = to_var(torch.LongTensor(target_utterances))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))

            sentence_logits, kl_div = self.model(utterances, utterance_length,
                                                 input_conversation_length, target_utterances)

            recon_loss, n_words = masked_cross_entropy(sentence_logits, target_utterances, target_utterance_length)
            batch_loss = recon_loss + kl_div

            batch_loss_history.append(batch_loss.item())
            recon_loss_history.append(recon_loss.item())
            kl_div_history.append(kl_div.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words
        epoch_recon_loss = np.sum(recon_loss_history) / n_total_words
        epoch_kl_div = np.sum(kl_div_history) / n_total_words

        print_str = f'Validation loss: {epoch_loss:.3f}, recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
        print(print_str)
        print('\n')

        return epoch_loss

    def test(self):
        raise NotImplementedError
