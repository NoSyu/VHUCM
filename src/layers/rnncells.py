import torch
import torch.nn as nn


class StackedGRUCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h):
        h_list = []
        for i, layer in enumerate(self.layers):
            h_i = layer(x, h[i])
            x = h_i
            if i + 1 is not self.num_layers:
                x = self.dropout(x)
            h_list.append(h_i)

        last_h = h_list[-1]
        h_list = torch.stack(h_list)

        return last_h, h_list


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(x, (h_0[i], c_0[i]))

            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return last_h_c, h_c_list


class LSTMSACell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMSACell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.u12u = nn.Linear(input_size, input_size, bias=False)
        self.u22u = nn.Linear(input_size, input_size, bias=False)
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.u2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, word_input, user1_input, user2_input, hidden):
        hx, cx = hidden

        user_gate = torch.tanh(self.u12u(user1_input) + self.u22u(user2_input))
        gates = self.x2h(word_input) + self.h2h(hx) + self.u2h(user_gate)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))

        return hy, cy

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
