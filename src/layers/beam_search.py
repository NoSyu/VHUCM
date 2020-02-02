import torch
from utils import EOS_ID


class Beam(object):
    def __init__(self, batch_size, hidden_size, vocab_size, beam_size, max_unroll, batch_position):
        """Beam class for beam search"""
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.max_unroll = max_unroll

        self.batch_position = batch_position

        self.log_probs = list()
        self.scores = list()
        self.back_pointers = list()
        self.token_ids = list()

        self.metadata = {'inputs': None, 'output': None, 'scores': None, 'length': None, 'sequence': None}

    def update(self, score, back_pointer, token_id):
        self.scores.append(score)
        self.back_pointers.append(back_pointer)
        self.token_ids.append(token_id)

    def backtrack(self):
        prediction = list()

        length = [[self.max_unroll] * self.beam_size for _ in range(self.batch_size)]

        top_k_score, top_k_idx = self.scores[-1].topk(self.beam_size, dim=1)

        top_k_score = top_k_score.clone()

        n_eos_in_batch = [0] * self.batch_size

        back_pointer = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)

        for t in reversed(range(self.max_unroll)):
            token_id = self.token_ids[t].index_select(0, back_pointer)
            back_pointer = self.back_pointers[t].index_select(0, back_pointer)
            eos_indices = self.token_ids[t].data.eq(EOS_ID).nonzero()

            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    eos_idx = eos_indices[i, 0].item()

                    batch_idx = eos_idx // self.beam_size
                    batch_start_idx = batch_idx * self.beam_size

                    _n_eos_in_batch = n_eos_in_batch[batch_idx] % self.beam_size
                    beam_idx_to_be_replaced = self.beam_size - _n_eos_in_batch - 1
                    idx_to_be_replaced = batch_start_idx + beam_idx_to_be_replaced

                    back_pointer[idx_to_be_replaced] = self.back_pointers[t][eos_idx].item()
                    token_id[idx_to_be_replaced] = self.token_ids[t][eos_idx].item()
                    top_k_score[batch_idx,
                                beam_idx_to_be_replaced] = self.scores[t].view(-1)[eos_idx].item()
                    length[batch_idx][beam_idx_to_be_replaced] = t + 1

                    n_eos_in_batch[batch_idx] += 1

            prediction.append(token_id)

        top_k_score, top_k_idx = top_k_score.topk(self.beam_size, dim=1)
        final_score = top_k_score.data

        for batch_idx in range(self.batch_size):
            length[batch_idx] = [length[batch_idx][beam_idx.item()]
                                 for beam_idx in top_k_idx[batch_idx]]

        top_k_idx = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)

        prediction = [step.index_select(0, top_k_idx).view(
            self.batch_size, self.beam_size) for step in reversed(prediction)]

        prediction = torch.stack(prediction, 2)

        return prediction, final_score, length
