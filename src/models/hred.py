import torch
import torch.nn as nn
from utils import to_var, pad
import layers


class HRED(nn.Module):
    def __init__(self, config):
        super(HRED, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size, config.embedding_size, config.encoder_hidden_size,
                                         config.rnn, config.num_layers, config.bidirectional, config.dropout,
                                         pretrained_wv_path=config.pretrained_wv_path)

        context_input_size = (config.num_layers * config.encoder_hidden_size * self.encoder.num_directions)
        self.context_encoder = layers.ContextRNN(context_input_size, config.context_size, config.rnn,
                                                 config.num_layers, config.dropout)

        self.decoder = layers.DecoderRNN(config.vocab_size, config.embedding_size, config.decoder_hidden_size,
                                         config.rnncell, config.num_layers, config.dropout, config.word_drop,
                                         config.max_unroll, config.sample, config.temperature, config.beam_size)

        self.context2decoder = layers.FeedForward(config.context_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1, activation=config.activation)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self, input_utterances, input_utterance_length, input_conversation_length,
                target_utterances, decode=False):
        """
        Forward of HRED
        :param input_utterances: [num_utterances, max_utter_len]
        :param input_utterance_length: [num_utterances]
        :param input_conversation_length: [batch_size]
        :param target_utterances: [num_utterances, seq_len]
        :param decode: True or False
        :return: decoder_outputs
        """
        num_utterances = input_utterances.size(0)
        max_conv_len = input_conversation_length.data.max().item()

        encoder_outputs, encoder_hidden = self.encoder(input_utterances, input_utterance_length)
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_utterances, -1)
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1])), 0)

        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l), max_conv_len)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden, input_conversation_length)
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        decoder_init = self.context2decoder(context_outputs)
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        if not decode:
            decoder_outputs = self.decoder(target_utterances, init_h=decoder_init, decode=decode)
            return decoder_outputs

        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            return prediction

    def generate(self, context, utterances_length, n_context):
        """
        Generate the response based on the context
        :param context: [batch_size, n_context, max_utter_len] given conversation utterances
        :param utterances_length: [batch_size, n_context] length of the utterances in the context
        :param n_context: length of the context turns
        :return: generated responses
        """
        batch_size = context.size(0)
        samples = []
        all_samples = list()

        context_hidden = None
        context_outputs = None
        for i in range(n_context):
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :], utterances_length[:, i])

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden, context_hidden)

        for j in range(self.config.n_sample_step):
            context_outputs = context_outputs.squeeze(1)
            decoder_init = self.context2decoder(context_outputs)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

            prediction_all, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            all_samples.append(prediction_all)
            prediction = prediction_all[:, 0, :]
            length = [l[0] for l in length]
            length = to_var(torch.LongTensor(length))
            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction, length)
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden, context_hidden)

        samples = torch.stack(samples, 1)
        all_samples = torch.stack(all_samples, 1)

        return samples, all_samples
