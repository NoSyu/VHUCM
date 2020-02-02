import numpy as np
import torch
from layers import masked_cross_entropy
from utils import to_var
import os
from tqdm import tqdm
from math import isnan
import codecs
from .solver import Solver


class SolverVHUCM(Solver):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SolverVHUCM, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)

    def train(self):
        epoch_loss_history = []
        kl_mult = 0.0
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            recon_loss_history = []
            kl_div_history = []
            bow_loss_history = []
            self.model.train()
            n_total_words = 0

            for batch_i, (conversations, users, conversation_length, sentence_length) \
                    in enumerate(tqdm(self.train_data_loader, ncols=80)):
                target_conversations = [conv[1:] for conv in conversations]

                conv_users = [list(set(conv_users)) for conv_users in users]
                input_users = [one_user for conv_users in users for one_user in conv_users]

                sentences = [sent for conv in conversations for sent in conv]
                input_conversation_length = [l - 1 for l in conversation_length]
                target_sentences = [sent for conv in target_conversations for sent in conv]
                target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
                sentence_length = [l for len_list in sentence_length for l in len_list]

                sentences = to_var(torch.LongTensor(sentences))
                conv_users = to_var(torch.LongTensor(conv_users))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
                input_users = to_var(torch.LongTensor(input_users))

                self.optimizer.zero_grad()

                sentence_logits, kl_div, _, _ = self.model(sentences, conv_users, input_users, sentence_length,
                                                           input_conversation_length, target_sentences)

                recon_loss, n_words = masked_cross_entropy(sentence_logits, target_sentences, target_sentence_length)

                batch_loss = recon_loss + kl_mult * kl_div
                batch_loss_history.append(batch_loss.item())
                recon_loss_history.append(recon_loss.item())
                kl_div_history.append(kl_div.item())
                n_total_words += n_words.item()

                if self.config.bow:
                    bow_loss = self.model.compute_bow_loss(target_conversations)
                    batch_loss += bow_loss
                    bow_loss_history.append(bow_loss.item())

                assert not isnan(batch_loss.item())

                if batch_i % self.config.print_every == 0:
                    print_str = f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item() / n_words.item():.3f}, recon = {recon_loss.item() / n_words.item():.3f}, kl_div = {kl_div.item() / n_words.item():.3f}'
                    if self.config.bow:
                        print_str += f', bow_loss = {bow_loss.item() / n_words.item():.3f}'
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

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}, recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
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

        return epoch_loss_history

    def evaluate(self):
        self.model.eval()
        batch_loss_history = []
        recon_loss_history = []
        kl_div_history = []
        bow_loss_history = []
        n_total_words = 0
        for batch_i, (conversations, users, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            target_conversations = [conv[1:] for conv in conversations]

            conv_users = [list(set(conv_users)) for conv_users in users]
            input_users = [one_user for conv_users in users for one_user in conv_users]

            sentences = [sent for conv in conversations for sent in conv]
            input_conversation_length = [l - 1 for l in conversation_length]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            sentence_length = [l for len_list in sentence_length for l in len_list]

            with torch.no_grad():
                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
                conv_users = to_var(torch.LongTensor(conv_users))
                input_users = to_var(torch.LongTensor(input_users))

            sentence_logits, kl_div, _, _ = self.model(sentences, conv_users, input_users, sentence_length,
                                                       input_conversation_length, target_sentences)

            recon_loss, n_words = masked_cross_entropy(sentence_logits, target_sentences, target_sentence_length)

            batch_loss = recon_loss + kl_div
            if self.config.bow:
                bow_loss = self.model.compute_bow_loss(target_conversations)
                bow_loss_history.append(bow_loss.item())

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            recon_loss_history.append(recon_loss.item())
            kl_div_history.append(kl_div.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words
        epoch_recon_loss = np.sum(recon_loss_history) / n_total_words
        epoch_kl_div = np.sum(kl_div_history) / n_total_words

        print_str = f'Validation loss: {epoch_loss:.3f}, recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
        if bow_loss_history:
            epoch_bow_loss = np.sum(bow_loss_history) / n_total_words
            print_str += f', bow_loss = {epoch_bow_loss:.3f}'
        print(print_str)
        print('\n')

        return epoch_loss

    def export_samples(self, beam_size=5):
        self.model.decoder.beam_size = beam_size
        self.model.eval()
        n_context = self.config.n_context
        n_sample_step = self.config.n_sample_step
        context_history = []
        sample_history = []
        ground_truth_history = list()

        for batch_i, (conversations, users, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            n_context_sample_step = n_context + n_sample_step
            conv_indices = [i for i in range(len(conversations)) if len(conversations[i]) >= n_context_sample_step]
            context = [c for i in conv_indices for c in [conversations[i][:n_context]]]
            ground_truth = [c for i in conv_indices for c in [conversations[i][n_context:n_context_sample_step]]]
            sentence_length = [c for i in conv_indices for c in [sentence_length[i][:n_context]]]
            context_users = [list(set(utter_users)) for i in conv_indices for utter_users in [users[i][:n_context_sample_step]]]
            input_users = [utter_users for i in conv_indices for utter_users in [users[i][:n_context_sample_step]]]

            with torch.no_grad():
                context = to_var(torch.LongTensor(context))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                context_users = to_var(torch.LongTensor(context_users))
                input_users = to_var(torch.LongTensor(input_users))

            _, all_samples = self.model.generate(context, context_users, input_users, sentence_length, n_context)

            context = context.data.cpu().numpy().tolist()
            all_samples = all_samples.data.cpu().numpy().tolist()
            context_history.append(context)
            sample_history.append(all_samples)
            ground_truth_history.append(ground_truth)

        target_file_name = 'samples_{}_{}_{}_{}.txt'.format(self.config.mode, n_context, beam_size, self.epoch_i)
        print("Writing candidates into file {}".format(target_file_name))
        conv_idx = 0
        with codecs.open(os.path.join(self.config.save_path, target_file_name), 'w', "utf-8") as output_f:
            for contexts, samples, ground_truths in tqdm(zip(context_history, sample_history, ground_truth_history),
                                                         total=len(context_history), ncols=80):
                for one_conv_contexts, one_conv_samples, one_conv_ground_truth in zip(contexts, samples, ground_truths):
                    print("Conversation Context {}".format(conv_idx), file=output_f)
                    print("\n".join([self.vocab.decode(utter) for utter in one_conv_contexts]), file=output_f)
                    print("\n".join([self.vocab.decode(utter) for utters_beam in one_conv_samples for utter in utters_beam]), file=output_f)
                    print("\n".join([self.vocab.decode(utter) for utter in one_conv_ground_truth]), file=output_f)
                    conv_idx += 1

        return conv_idx
