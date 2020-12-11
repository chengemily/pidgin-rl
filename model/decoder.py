import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, rnn_type='GRU', nlayers=1, dropout=0., bidirectional=True, batch_first=True):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        rnn_cell = getattr(nn, rnn_type) # get constructor from torch.nn
        self.batch_first = batch_first
        self.rnn = rnn_cell(
            embed_dim, hidden_dim, nlayers, dropout=dropout, bidirectional=bidirectional, batch_first=True
        )

    def forward(self, inp, x_lengths, hidden=None):
        # print(x_lengths)
        # Pack padded sequence so padded items aren't shown to LSTM-- perf increase

        total_length = inp.size()[1]

        X = torch.nn.utils.rnn.pack_padded_sequence(inp, x_lengths, batch_first=self.batch_first, enforce_sorted=False)
        # print("Packed 1st sentence:", X[0])
        # print("X: (CONFIRM BATCH SIZES)", X)
        out, hid = self.rnn(X, hidden) # since it's packed, "hid" is the last meaningful hidden state
        # print("Packed out: ", out[0])
        # input(hid[0].size()) # should be 2 x 32 x 50
        # Unpack output
        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, total_length=total_length, batch_first=self.batch_first, padding_value=0.0)
        # print("Packed first sentence: ", out[0])
        # input()
        return out, hid


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.att_scores = None

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK],
        # Values = [BxTxV], should be the outputs of the RNN
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        # print("Q size: ", query.size())
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        # print("K size: ", keys.size())
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
        self.att_scores = energy

        # values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class FC(nn.Module):
    def __init__(self, layer_dims=[]):
        super(FC, self).__init__()
        layer_dims.append(2) # last output size 2 vector
        self.layers = [nn.Linear(layer_dims[i], layer_dims[i+1]).cuda() for i in range(len(layer_dims)-1)]

    def forward(self, input):
        inp = input
        for layer in self.layers:
            inp = layer(inp)
        return torch.tanh(inp)


class Vectorizer(nn.Module):
    def __init__(self, embedding, encoder, fc_layers, attention, concat_out=False, use_attn=False, end2end=False):
        """
        Entire str -> vector model.
        :param embedding: embedder class, where embedding(input) returns a vector
        :param encoder: (nn.Module)
        :param fc_layers: (list(int)) dimensions of fc layers not including the output of 2
        :param attention: (nn.Module) attention class
        """
        super(Vectorizer, self).__init__()
        self.end2end = end2end
        self.embedding = embedding if not self.end2end else self.make_embedding(embedding)
        self.encoder = encoder
        self.attention = attention
        self.concat_out = concat_out
        self.use_attn = use_attn
        self.fc = FC(fc_layers) if not isinstance(fc_layers, nn.Module) else fc_layers

        param_size = sum([p.nelement() for p in self.parameters()])
        print('Total param size: {}'.format(param_size))

    def make_embedding(self, embedding):
        """
        Creates an embedding layer depending on whether the module is end2end.
        :param embedding: (nn.Embedding)
        :return: an nn linear layer with the weights as in nn.embedding
        """
        weights = embedding.weight  # V x emsize
        new_embed = torch.nn.Linear(weights.size()[0], weights.size()[1])
        with torch.no_grad():
            new_embed.weight.copy_(torch.transpose(weights, 0, 1))

        return new_embed

    def forward(self, input, x_lengths):

        outputs, hidden = self.encoder(self.embedding(input), x_lengths)

        if isinstance(hidden, tuple): # LSTM
          hidden = hidden[1] # take the cell state
        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
          hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
          hidden = hidden[-1]

        # fc_input = torch.cat([hidden, outputs], dim=1)
        energy=None
        if self.concat_out:
            # concatenate output vectors
            outputs = outputs.reshape(outputs.size()[0], outputs.size()[1] * outputs.size()[2])
            output_vec = self.fc(outputs)
        elif self.use_attn:
            energy, linear_combination = self.attention(hidden, outputs, outputs)
            output_vec = self.fc(linear_combination)
        else:
            output_vec = self.fc(hidden)

        # Activation
        return output_vec, energy
