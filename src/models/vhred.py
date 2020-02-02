import torch
import torch.nn as nn
from utils import to_var, pad, normal_kl_div
import layers


class VHRED(nn.Module):
    def __init__(self, config):
        super(VHRED, self).__init__()

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

        self.context2decoder = layers.FeedForward(config.context_size + config.z_utter_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1, activation=config.activation)

        self.softplus = nn.Softplus()
        self.prior_h = layers.FeedForward(config.context_size, config.context_size, num_layers=2,
                                          hidden_size=config.context_size, activation=config.activation)
        self.prior_mu = nn.Linear(config.context_size, config.z_utter_size)
        self.prior_var = nn.Linear(config.context_size, config.z_utter_size)

        self.posterior_h = layers.FeedForward(config.encoder_hidden_size * self.encoder.num_directions * config.num_layers + config.context_size,
                                              config.context_size, num_layers=2, hidden_size=config.context_size,
                                              activation=config.activation)
        self.posterior_mu = nn.Linear(config.context_size, config.z_utter_size)
        self.posterior_var = nn.Linear(config.context_size, config.z_utter_size)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def prior(self, context_outputs):
        """
        Compute prior
        :param context_outputs: h_t^{cxt} [num_true_utterances, context_rnn_output_size]
        :return: [mu, sigma]
        """
        pass

    def posterior(self, context_outputs, encoder_hidden):
        """
        Compute variational posterior
        :param context_outputs: h_t^{cxt} [num_true_utterances, context_rnn_output_size]
        :param encoder_hidden: x_t [num_true_utterances, encoder_rnn_output_size]
        :return: [mu, sigma]
        """
        pass

    def forward(self, utterances, utterance_length, input_conversation_length, target_utterances,
                decode=False):
        """
        Forward of VHRED
        :param utterances: [num_utterances, max_utter_len]
        :param utterance_length: [num_utterances]
        :param input_conversation_length: [batch_size]
        :param target_utterances: [num_utterances, seq_len]
        :param decode: True or False
        :return: decoder_outputs
        """
        batch_size = input_conversation_length.size(0)
        num_utterances = utterances.size(0)
        max_conv_len = input_conversation_length.data.max().item()

        encoder_outputs, encoder_hidden = self.encoder(utterances, utterance_length)
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_utterances, -1)
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1] + 1)), 0)

        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1), max_conv_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat([encoder_hidden_inference[i, :l, :]
                                                   for i, l in enumerate(input_conversation_length.data)])

        encoder_hidden_input = encoder_hidden[:, :-1, :]

        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden_input, input_conversation_length)
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        mu_prior, var_prior = self.prior(context_outputs)
        eps = to_var(torch.randn((num_utterances - batch_size, self.config.z_utter_size)))
        if not decode:
            mu_posterior, var_posterior = self.posterior(context_outputs, encoder_hidden_inference_flat)
            z_sent = mu_posterior + torch.sqrt(var_posterior) * eps

            kl_div = normal_kl_div(mu_posterior, var_posterior, mu_prior, var_prior)
            kl_div = torch.sum(kl_div)
        else:
            z_sent = mu_prior + torch.sqrt(var_prior) * eps
            kl_div = None

        self.z_sent = z_sent
        latent_context = torch.cat([context_outputs, z_sent], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1, self.decoder.num_layers, self.decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()

        if not decode:
            decoder_outputs = self.decoder(target_utterances, init_h=decoder_init, decode=decode)
            return decoder_outputs, kl_div
        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            return prediction, kl_div

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

        context_hidden=None
        for i in range(n_context):
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :], utterances_length[:, i])

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden, context_hidden)

        for j in range(self.config.n_sample_step):
            context_outputs = context_outputs.squeeze(1)

            mu_prior, var_prior = self.prior(context_outputs)
            eps = to_var(torch.randn((batch_size, self.config.z_utter_size)))
            z_sent = mu_prior + torch.sqrt(var_prior) * eps

            latent_context = torch.cat([context_outputs, z_sent], 1)
            decoder_init = self.context2decoder(latent_context)
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
