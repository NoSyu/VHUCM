import random
import torch
from torch import nn
from torch.nn import functional as F
from .rnncells import StackedGRUCell, LSTMSACell
from .beam_search import Beam
from utils import to_var, SOS_ID, UNK_ID, EOS_ID


class BaseRNNDecoder(nn.Module):
    def __init__(self):
        super(BaseRNNDecoder, self).__init__()

    def init_token(self, batch_size):
        x = to_var(torch.LongTensor([SOS_ID] * batch_size))
        return x

    def init_h(self, batch_size=None, zero=True, hidden=None):
        if hidden is not None:
            return hidden

        return to_var(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size
        else:
            batch_size = h.size(1)
            return batch_size

    def decode(self, out):
        if self.sample:
            x = torch.multinomial(self.softmax(out / self.temperature), 1).view(-1)
        else:
            _, x = out.max(dim=1)
        return x

    def forward(self):
        raise NotImplementedError

    def forward_step(self):
        raise NotImplementedError

    def embed(self, x):
        if self.training and self.word_drop > 0.0:
            if random.random() < self.word_drop:
                embed = self.embedding(to_var(x.data.new([UNK_ID] * x.size(0))))
            else:
                embed = self.embedding(x)
        else:
            embed = self.embedding(x)

        return embed

    def beam_decode(self, init_h=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(h=init_h)

        x = self.init_token(batch_size * self.beam_size)

        h = self.init_h(batch_size, hidden=init_h).repeat(1, self.beam_size, 1)

        batch_position = to_var(torch.arange(0, batch_size).long() * self.beam_size)

        score = torch.ones(batch_size * self.beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * self.beam_size, 0.0)
        score = to_var(score)

        beam = Beam(batch_size, self.hidden_size, self.vocab_size, self.beam_size, self.max_unroll, batch_position)

        for i in range(self.max_unroll):
            out, h = self.forward_step(x, h, encoder_outputs=encoder_outputs, input_valid_length=input_valid_length)
            log_prob = F.log_softmax(out, dim=1)

            score = score.view(-1, 1) + log_prob

            score, top_k_idx = score.view(batch_size, -1).topk(self.beam_size, dim=1)

            x = (top_k_idx % self.vocab_size).view(-1)

            beam_idx = top_k_idx / self.vocab_size  # [batch_size, beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            h = h.index_select(1, top_k_pointer)

            beam.update(score.clone(), top_k_pointer, x)  # , h)

            eos_idx = x.data.eq(EOS_ID).view(batch_size, self.beam_size)
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length


class DecoderRNN(BaseRNNDecoder):
    def __init__(self, vocab_size, embedding_size, hidden_size, rnncell=StackedGRUCell, num_layers=1,
                 dropout=0.0, word_drop=0.0, max_unroll=30, sample=True, temperature=1.0, beam_size=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.word_drop = word_drop
        self.max_unroll = max_unroll
        self.sample = sample
        self.beam_size = beam_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.rnncell = rnncell(num_layers, embedding_size, hidden_size, dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward_step(self, x, h, encoder_outputs=None, input_valid_length=None):
        x = self.embed(x)
        last_h, h = self.rnncell(x, h)
        out = self.out(last_h)

        return out, h

    def forward(self, inputs, init_h=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(inputs, init_h)

        x = self.init_token(batch_size)

        h = init_h

        if not decode:
            out_list = []
            seq_len = inputs.size(1)
            for i in range(seq_len):
                out, h = self.forward_step(x, h)

                out_list.append(out)
                x = inputs[:, i]

            return torch.stack(out_list, dim=1)
        else:
            x_list = []
            for i in range(self.max_unroll):
                out, h = self.forward_step(x, h)

                x = self.decode(out)
                x_list.append(x)

            return torch.stack(x_list, dim=1)


class DecoderSARNN(BaseRNNDecoder):
    def __init__(self, vocab_size, user_size, embedding_size, hidden_size, num_layers=1,
                 dropout=0.0, word_drop=0.0, max_unroll=30, sample=True, temperature=1.0, beam_size=1):
        super(DecoderSARNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.word_drop = word_drop
        self.max_unroll = max_unroll
        self.sample = sample
        self.beam_size = beam_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.user_embedding = nn.Embedding(user_size, embedding_size)

        self.rnncell = LSTMSACell(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward_step(self, x, user1_embed, user2_embed, h, encoder_outputs=None, input_valid_length=None):
        x = self.embed(x)
        hy, cy = self.rnncell(x, user1_embed, user2_embed, h)
        last_h = hy
        out = self.out(last_h)

        return out, (hy, cy)

    def forward(self, word_inputs, user_inputs, init_h=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(word_inputs, init_h)
        user_embedded = self.user_embedding(user_inputs)

        x = self.init_token(batch_size)
        h = init_h

        if not decode:
            out_list = []
            seq_len = word_inputs.size(1)
            for i in range(seq_len):
                out, h = self.forward_step(x, user_embedded[:, 0, :], user_embedded[:, 1, :], h)

                out_list.append(out)
                x = word_inputs[:, i]

            return torch.stack(out_list, dim=1)
        else:
            x_list = []
            for i in range(self.max_unroll):
                out, h = self.forward_step(x, user_embedded[:, 0, :], user_embedded[:, 1, :], h)

                x = self.decode(out)
                x_list.append(x)

            return torch.stack(x_list, dim=1)

    def beam_decode(self, init_h=None, user_inputs=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(user_inputs, h=init_h)
        user_embedded = self.user_embedding(user_inputs)

        x = self.init_token(batch_size * self.beam_size)

        hx, cx = init_h
        hx = hx.repeat(self.beam_size, 1)
        cx = cx.repeat(self.beam_size, 1)
        h = (hx, cx)
        user_emb_0 = user_embedded[:, 0, :].repeat(self.beam_size, 1)
        user_emb_1 = user_embedded[:, 1, :].repeat(self.beam_size, 1)

        batch_position = to_var(torch.arange(0, batch_size).long() * self.beam_size)

        score = torch.ones(batch_size * self.beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * self.beam_size, 0.0)
        score = to_var(score)

        beam = Beam(batch_size, self.hidden_size, self.vocab_size, self.beam_size, self.max_unroll, batch_position)

        for i in range(self.max_unroll):
            out, h = self.forward_step(x, user_emb_0, user_emb_1, h,
                                       encoder_outputs=encoder_outputs, input_valid_length=input_valid_length)
            log_prob = F.log_softmax(out, dim=1)

            score = score.view(-1, 1) + log_prob

            score, top_k_idx = score.view(batch_size, -1).topk(self.beam_size, dim=1)

            x = (top_k_idx % self.vocab_size).view(-1)

            beam_idx = top_k_idx / self.vocab_size
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            h = (h[0].index_select(0, top_k_pointer), h[1].index_select(0, top_k_pointer))
            # h = h.index_select(1, top_k_pointer)

            beam.update(score.clone(), top_k_pointer, x)

            eos_idx = x.data.eq(EOS_ID).view(batch_size, self.beam_size)
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length
