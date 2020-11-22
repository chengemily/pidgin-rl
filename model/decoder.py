import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, rnn_type='GRU', nlayers=1, dropout=0., bidirectional=True, batch_first=True):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        rnn_cell = getattr(nn, rnn_type) # get constructor from torch.nn
        self.rnn = rnn_cell(
            embed_dim, hidden_dim, nlayers, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first
        )

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


# class LuongAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(LuongAttention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.att_scores = None
#
#     def forward(self, decoder_hidden, encoder_outputs):
#         # project decoder hidden state
#         out = self.W(decoder_hidden)
#
#         # Calculate new outputs
#         alignments = encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)
#
#         # Store attention scores
#         atts = F.softmax(alignments.view(1, -1), dim=1)
#         self.att_scores = atts
#
#         # Get new context vector
#         context = torch.bmm(atts.unsqueeze(0), encoder_outputs)
#
#         return atts, context

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.att_scores = None

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
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
        self.layers = [nn.Linear(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]

    def forward(self, input):
        inp = input
        for layer in self.layers:
            inp = layer(inp)
        return inp


class Vectorizer(nn.Module):
    def __init__(self, embedding, encoder, fc_layer_dims, attention):
        """
        Entire str -> vector model.
        :param embedding: embedder class, where embedding(input) returns a vector
        :param encoder: (nn.Module)
        :param decoder_layer_dims: (list(int)) dimensions of decoder layer not including the output of 2
        :param attention: (nn.Module) attention class
        """
        super(Vectorizer, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.fc = FC(fc_layer_dims)

        param_size = sum([p.nelement() for p in self.parameters()])
        print('Total param size: {}'.format(param_size))

    def forward(self, input):
        outputs, hidden = self.encoder(self.embedding(input))

        # concatenate outputs vectors
        # outputs = outputs.reshape(outputs.size()[0], outputs.size()[1] * outputs.size()[2])

        if isinstance(hidden, tuple): # LSTM
          hidden = hidden[1] # take the cell state

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
          hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
          hidden = hidden[-1]

        energy=None
        # output_vec = self.fc(outputs)
        # energy, linear_combination = self.attention(hidden, outputs, outputs)
        # output_vec = self.fc(linear_combination)
        output_vec = self.fc(hidden)
        return output_vec, energy
