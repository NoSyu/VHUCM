import torch
import torch.nn as nn
from utils import to_var, pad, normal_kl_div, normal_logpdf, bag_of_words_loss, to_bow, EOS_ID
import layers
import numpy as np
import pickle


class VHUCM(nn.Module):
    def __init__(self, config):
        super(VHUCM, self).__init__()

        self.config = config

        # user embedding
        if config.pretrained_uv_path is None:
            self.user_embedding = nn.Embedding(config.user_size, config.embedding_size)
            self.user_embedding.weight.requires_grad = True
        else:
            with open(config.pretrained_uv_path, 'rb') as f:
                weight_tensor = to_var(torch.FloatTensor(pickle.load(f)))

            self.user_embedding = nn.Embedding.from_pretrained(weight_tensor, freeze=False)

        self.encoder = layers.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         config.bidirectional,
                                         config.dropout,
                                         pretrained_wv_path=config.pretrained_wv_path)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions + config.z_conv_size + config.embedding_size)
        self.context_encoder = layers.ContextRNN(context_input_size,
                                                 config.context_size,
                                                 config.rnn,
                                                 config.num_layers,
                                                 config.dropout)

        self.unk_sent = nn.Parameter(torch.randn(context_input_size - (config.z_conv_size + config.embedding_size)))

        self.z_conv2context = layers.FeedForward(config.z_conv_size,
                                                 config.num_layers * config.context_size,
                                                 num_layers=1,
                                                 activation=config.activation)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)
        self.context_inference = layers.ContextRNN(context_input_size,
                                                   config.context_size,
                                                   config.rnn,
                                                   config.num_layers,
                                                   config.dropout,
                                                   bidirectional=True)

        self.decoder = layers.DecoderRNN(config.vocab_size,
                                        config.embedding_size,
                                        config.decoder_hidden_size,
                                        config.rnncell,
                                        config.num_layers,
                                        config.dropout,
                                        config.word_drop,
                                        config.max_unroll,
                                        config.sample,
                                        config.temperature,
                                        config.beam_size)

        self.context2decoder = layers.FeedForward(config.context_size + config.z_sent_size + config.z_conv_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)

        self.softplus = nn.Softplus()

        self.conv_prior_h = layers.FeedForward(2*config.embedding_size,
                                               config.embedding_size, num_layers=1, hidden_size=config.z_conv_size,
                                               activation=config.activation)
        self.conv_prior_mu = nn.Linear(config.embedding_size, config.z_conv_size)
        self.conv_prior_var = nn.Linear(config.embedding_size, config.z_conv_size)

        self.conv_posterior_h = layers.FeedForward(config.num_layers * self.context_inference.num_directions * config.context_size + 2*config.embedding_size,
                                                    config.context_size,
                                                    num_layers=2,
                                                    hidden_size=config.context_size,
                                                    activation=config.activation)
        self.conv_posterior_mu = nn.Linear(config.context_size,
                                            config.z_conv_size)
        self.conv_posterior_var = nn.Linear(config.context_size,
                                             config.z_conv_size)

        self.sent_prior_h = layers.FeedForward(config.context_size + config.z_conv_size,
                                               config.context_size,
                                               num_layers=1,
                                               hidden_size=config.z_sent_size,
                                               activation=config.activation)
        self.sent_prior_mu = nn.Linear(config.context_size,
                                       config.z_sent_size)
        self.sent_prior_var = nn.Linear(config.context_size,
                                        config.z_sent_size)

        self.sent_posterior_h = layers.FeedForward(config.z_conv_size + config.encoder_hidden_size * self.encoder.num_directions * config.num_layers + config.context_size,
                                                   config.context_size,
                                                   num_layers=2,
                                                   hidden_size=config.context_size,
                                                   activation=config.activation)
        self.sent_posterior_mu = nn.Linear(config.context_size,
                                           config.z_sent_size)
        self.sent_posterior_var = nn.Linear(config.context_size,
                                            config.z_sent_size)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

        if config.bow:
            self.bow_h = layers.FeedForward(config.z_sent_size,
                                            config.decoder_hidden_size,
                                            num_layers=1,
                                            hidden_size=config.decoder_hidden_size,
                                            activation=config.activation)
            self.bow_predict = nn.Linear(config.decoder_hidden_size, config.vocab_size)

    def conv_prior(self, user1_emb, user2_emb):
        h_prior = self.conv_prior_h(torch.cat([user1_emb, user2_emb], dim=1))
        mu_prior = self.conv_prior_mu(h_prior)
        var_prior = self.softplus(self.conv_prior_var(h_prior))
        return mu_prior, var_prior

    def conv_posterior(self, context_inference_hidden, user1_emb, user2_emb):
        h_posterior = self.conv_posterior_h(torch.cat([context_inference_hidden, user1_emb, user2_emb], dim=1))
        mu_posterior = self.conv_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.conv_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def sent_prior(self, context_outputs, z_conv):
        # Context dependent prior
        h_prior = self.sent_prior_h(torch.cat([context_outputs, z_conv], dim=1))
        mu_prior = self.sent_prior_mu(h_prior)
        var_prior = self.softplus(self.sent_prior_var(h_prior))
        return mu_prior, var_prior

    def sent_posterior(self, context_outputs, encoder_hidden, z_conv):
        h_posterior = self.sent_posterior_h(torch.cat([context_outputs, encoder_hidden, z_conv], 1))
        mu_posterior = self.sent_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.sent_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def forward(self, sentences, conv_users, users, sentence_length,
                input_conversation_length, target_sentences, decode=False):
        batch_size = input_conversation_length.size(0)
        num_sentences = sentences.size(0) - batch_size
        max_len = input_conversation_length.data.max().item()

        encoder_outputs, encoder_hidden = self.encoder(sentences, sentence_length)

        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_sentences + batch_size, -1)

        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1] + 1)), 0)
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1), max_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat([encoder_hidden_inference[i, :l, :] for i, l
                                                   in enumerate(input_conversation_length.data)])

        encoder_hidden_input = encoder_hidden[:, :-1, :]

        embedded_users = self.user_embedding(users)
        embedded_users = torch.stack([pad(embedded_users.narrow(0, s, l + 1), max_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)
        embedded_users_input = embedded_users[:, :-1, :]

        user_embedded = self.user_embedding(conv_users)
        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        conv_mu_prior, conv_var_prior = self.conv_prior(user_embedded[:, 0, :], user_embedded[:, 1, :])

        if not decode:
            if self.config.sentence_drop > 0.0:
                indices = np.where(np.random.rand(max_len) < self.config.sentence_drop)[0]
                if len(indices) > 0:
                    encoder_hidden_input[:, indices, :] = self.unk_sent

            context_inference_outputs, context_inference_hidden = self.context_inference(encoder_hidden,
                                                                                         input_conversation_length + 1)

            context_inference_hidden = context_inference_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden,
                                                                        user_embedded[:, 0, :], user_embedded[:, 1, :])

            z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps
            log_q_zx_conv = normal_logpdf(z_conv, conv_mu_posterior, conv_var_posterior).sum()

            log_p_z_conv = normal_logpdf(z_conv, conv_mu_prior, conv_var_prior).sum()
            kl_div_conv = normal_kl_div(conv_mu_posterior, conv_var_posterior, conv_mu_prior, conv_var_prior).sum()

            context_init = self.z_conv2context(z_conv).view(self.config.num_layers, batch_size, self.config.context_size)

            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(1)).expand(z_conv.size(0), max_len, z_conv.size(1))
            context_outputs, context_last_hidden = \
                self.context_encoder(torch.cat([encoder_hidden_input, z_conv_expand, embedded_users_input], 2),
                                     input_conversation_length, hidden=context_init)

            context_outputs = torch.cat([context_outputs[i, :l, :]
                                         for i, l in enumerate(input_conversation_length.data)])

            z_conv_flat = torch.cat([z_conv_expand[i, :l, :] for i, l in enumerate(input_conversation_length.data)])
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))

            sent_mu_posterior, sent_var_posterior = self.sent_posterior(context_outputs, encoder_hidden_inference_flat,
                                                                        z_conv_flat)
            z_sent = sent_mu_posterior + torch.sqrt(sent_var_posterior) * eps
            log_q_zx_sent = normal_logpdf(z_sent, sent_mu_posterior, sent_var_posterior).sum()

            log_p_z_sent = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum()
            kl_div_sent = normal_kl_div(sent_mu_posterior, sent_var_posterior, sent_mu_prior, sent_var_prior).sum()

            kl_div = kl_div_conv + kl_div_sent
            log_q_zx = log_q_zx_conv + log_q_zx_sent
            log_p_z = log_p_z_conv + log_p_z_sent
        else:
            z_conv = conv_mu_prior + torch.sqrt(conv_var_prior) * conv_eps
            context_init = self.z_conv2context(z_conv).view(self.config.num_layers, batch_size, self.config.context_size)

            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(1)).expand(z_conv.size(0), max_len, z_conv.size(1))
            context_outputs, context_last_hidden = \
                self.context_encoder(torch.cat([encoder_hidden_input, z_conv_expand, embedded_users_input], 2),
                                     input_conversation_length, hidden=context_init)
            context_outputs = torch.cat([context_outputs[i, :l, :]
                                         for i, l in enumerate(input_conversation_length.data)])

            z_conv_flat = torch.cat([z_conv_expand[i, :l, :] for i, l in enumerate(input_conversation_length.data)])
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))

            z_sent = sent_mu_prior + torch.sqrt(sent_var_prior) * eps
            kl_div = None
            log_p_z = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum()
            log_p_z += normal_logpdf(z_conv, conv_mu_prior, conv_var_prior).sum()
            log_q_zx = None

        self.z_sent = z_sent

        z_conv = torch.cat([z.view(1, -1).expand(m.item(), self.config.z_conv_size)
                            for z, m in zip(z_conv, input_conversation_length)])

        latent_context = torch.cat([context_outputs, z_sent, z_conv], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1, self.decoder.num_layers, self.decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()

        if not decode:
            decoder_outputs = self.decoder(target_sentences, init_h=decoder_init, decode=decode)
            return decoder_outputs, kl_div, log_p_z, log_q_zx

        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            return prediction, kl_div, log_p_z, log_q_zx

    def generate(self, context, context_users, utter_users, sentence_length, n_context):
        batch_size = context.size(0)
        samples = []
        all_samples = list()

        user_embedded = self.user_embedding(context_users)
        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))

        encoder_hidden_list = []
        for i in range(n_context):
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :], sentence_length[:, i])
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            encoder_hidden_list.append(encoder_hidden)

        encoder_hidden = torch.stack(encoder_hidden_list, 1)
        context_inference_outputs, context_inference_hidden = self.context_inference(encoder_hidden,
                                                                                     to_var(torch.LongTensor([n_context] * batch_size)))
        context_inference_hidden = context_inference_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
        conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden,
                                                                    user_embedded[:, 0, :], user_embedded[:, 1, :])
        z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps

        context_init = self.z_conv2context(z_conv).view(self.config.num_layers, batch_size, self.config.context_size)

        embedded_users = self.user_embedding(utter_users)

        context_hidden = context_init
        for i in range(n_context):
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :], sentence_length[:, i])
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            encoder_hidden_list.append(encoder_hidden)
            context_outputs, context_hidden = \
                self.context_encoder.step(torch.cat([encoder_hidden, z_conv, embedded_users[:, i, :]], 1),
                                          context_hidden)

        for j in range(self.config.n_sample_step):
            context_outputs = context_outputs.squeeze(1)

            mu_prior, var_prior = self.sent_prior(context_outputs, z_conv)
            eps = to_var(torch.randn((batch_size, self.config.z_sent_size)))
            z_sent = mu_prior + torch.sqrt(var_prior) * eps

            latent_context = torch.cat([context_outputs, z_sent, z_conv], 1)
            decoder_init = self.context2decoder(latent_context)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

            if self.config.sample:
                prediction = self.decoder(None, decoder_init, decode=True)
                p = prediction.data.cpu().numpy()
                length = torch.from_numpy(np.where(p == EOS_ID)[1])
            else:
                prediction_all, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
                all_samples.append(prediction_all)
                prediction = prediction_all[:, 0, :]
                length = [l[0] for l in length]
                length = to_var(torch.LongTensor(length))

            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction, length)

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

            context_outputs, context_hidden = \
                self.context_encoder.step(torch.cat([encoder_hidden, z_conv, embedded_users[:, n_context+j, :]], 1),
                                          context_hidden)

        samples = torch.stack(samples, 1)
        all_samples = torch.stack(all_samples, 1)

        return samples, all_samples

    def compute_bow_loss(self, target_conversations):
        target_bow = np.stack([to_bow(sent, self.config.vocab_size) for conv in target_conversations for sent in conv], axis=0)
        target_bow = to_var(torch.FloatTensor(target_bow))
        bow_logits = self.bow_predict(self.bow_h(self.z_sent))
        bow_loss = bag_of_words_loss(bow_logits, target_bow)
        return bow_loss
