import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# set global token values
SOS_token = 1
EOS_token = 2

# TODO - initialize word_ix_vec mapping - fix this in decoder/main.py
word_to_vec = {'<cls>': [-3.6961872577667236, -4.380651950836182, 0.8376801609992981, 0.4666217863559723, 8.50197696685791, 
                         -5.620670795440674, -2.2317745685577393, -3.4087929725646973, -5.6453375816345215, -0.7817500233650208]
               }

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, rnn_type='GRU', nlayers=1, dropout=0.):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, hidden_dim) # embedding matrix

        # Define recurrent unit
        self.rnn_type = rnn_type
        rnn_cell = getattr(nn, rnn_type) # get constructor from torch.nn
        self.rnn = rnn_cell(
            output_dim, hidden_dim, nlayers, dropout=dropout) # note input dim should be output_dim

        # Define linear and softmax units, assumes input of shape (batch, sentence_length, vector_length)
        self.out = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.LogSoftmax(dim=1) # dim=1 means take softmax across first dimension
        

    def forward(self, input=None, h0=None):
        if not input:
            input = self.init_output()
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, h0)
        if isinstance(hidden, tuple):
            hidden = hidden[1] # take the cell state if LSTM
        # output = self.softmax(self.out(output[0])) #TODO - why output[0]?
        output = self.out(output[0])
        return output, hidden

    # def init_output(self):
    #     # TODO - why initialize with zeros? How to initialize with <cls> token?
    #     # output = torch.zeros(1,1, self.hidden_dim) # TODO - actually, find vector representing cls
    #     # output[1] = 1
    #     # return output
    # attempt 2, but probably not even needed
    #     return torch.tensor(word_to_vec['<cls>'])



class FC_Encoder(nn.Module):
    def __init__(self, layer_dims=[]):
        super(FC_Encoder, self).__init__()
        layer_dims = [2] + layer_dims # input is size 2 vector
        self.layers = [nn.Linear(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
        # TODO - add activation functions between layers?

    def forward(self, input):
        inp = input
        for layer in self.layers:
            inp = layer(inp)
        return inp



class Sequence_Generator(nn.Module):
    def __init__(self, embedding, decoder, fc_layer_dims, hidden_dim, target_length):
        """
        Entire vector -> str model (the 'encoder' in our problem setup)
        :param embedding: embedder class, where embedding(input) returns a vector
        :param decoder: (nn.Module)
        :param fc_layer_dims: dimensions of fully connected layers, EXCLUDING hidden dim
        :param target_length: the maximum length a sequence can be (max seq length)
        """
        super(Sequence_Generator, self).__init__()
        self.target_length = target_length
        self.embedding = embedding
        self.decoder = decoder
        self.fc = FC_Encoder(fc_layer_dims + [hidden_dim])
        #
        # param_size = sum([p.nelement() for p in self.parameters()]) # TODO - not sure where self.parameters is coming from
        # print('Total param size: {}'.format(param_size))

    def forward(self, input): # Input should be [x,y] value
        '''
        :param input: should be a [x,y] pair
        :return:
        '''
        # get initial hidden and input
        init_hidden = self.fc(input)

        OUTPUTS = []

        # init decoder input and hidden
        decoder_input = torch.tensor(word_to_vec['<cls>']) #TODO - change word_to_vec to be 
        decoder_hidden = init_hidden

        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            OUTPUTS.append(decoder_output) # keep track of current output
            # from pytorch tutorial:
            # topv, topi = decoder_output.topk(1)
            # decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_input = decoder_output
            # if decoder_input.item() == EOS_token: # not doing anything right now
            #     break
        
        return torch.tensor(OUPUTS)




# QUESTION - do we need to modify attention for the decoder?
# class Attention(nn.Module):
#     def __init__(self, query_dim, key_dim, value_dim):
#         super(Attention, self).__init__()
#         self.scale = 1. / math.sqrt(query_dim)
#         self.att_scores = None
#
#     def forward(self, query, keys, values):
#         # Query = [BxQ]
#         # Keys = [TxBxK]
#         # Values = [TxBxV]
#         # Outputs = a:[TxB], lin_comb:[BxV]
#
#         # Here we assume q_dim == k_dim (dot product attention)
#
#         query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
#         keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
#         energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
#         energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
#         self.att_scores = energy
#
#         values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
#         linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
#         return energy, linear_combination



