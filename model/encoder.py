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
    def __init__(self, output_dim, hidden_dim, embedding, rnn_type='GRU', nlayers=1, dropout=0.):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)

        # Define recurrent unit
        self.rnn_type = rnn_type
        rnn_cell = getattr(nn, rnn_type) # get constructor from torch.nn
        self.rnn = rnn_cell(hidden_dim, # 1st param - input size, 2nd param - hidden size
                            hidden_dim,
                            nlayers, dropout=dropout,#,
                            batch_first=True)  # if inputs are (batch_size, seq, feature)

        # Define params needed for output unit
        # Define linear and softmax units, assumes input of shape (batch, sentence_length, vector_length)
        self.out = nn.Linear(hidden_dim, output_dim) # 1st param - size of input, 2nd param - size of output
        self.softmax = nn.LogSoftmax(dim=0) # dim=1 means take softmax across first dimension
        

    def forward(self, input, h0):
        # if LSTM, change hidden state

        output = self.embedding(input).view(-1, 1, self.hidden_dim) # TODO - added a third dimension using unsqueeze, check later if errors
        # print(f'embedded input: {output.shape}')
        output = self.dropout(output)
        
        # output = F.relu(output)
        
        # print(f'output shape: {output.size()}')
        
        # if isinstance(h0, tuple):
            # print(f'hidden shape: {h0[0].size()}')
        # else: print(f'hidden shape: {h0.size()}')

        # print(f'started rnn')
        output, hidden = self.rnn(output, h0) #TODO - make sure h0 is a tuple if using LSTM, one val if gru
        # print(f'output of RNN unit: {output}')
        # print(f'hidden output of RNN unit: {hidden}')

        # pass output through fcl and softmax
        output = self.out(output) # take output[0]?
        # print(f'after self.out : {output}')
        output = self.softmax(output).float.squeeze() #TODO - why output[0]?
        # print(f'after sotmax : {output}')
        return output, hidden




class FC_Encoder(nn.Module):
    def __init__(self, layer_dims=[]):
        super(FC_Encoder, self).__init__()
        layer_dims = [2] + layer_dims # input is size 2 vector
        # print(f'fcl layer dims : {layer_dims}')
        self.layers =  [nn.Linear(layer_dims[i], layer_dims[i+1]).cuda() for i in range(len(layer_dims)-1)]
        # print(self.layers)
        # TODO - add activation functions between layers?

    def forward(self, input):
        inp = input
        for layer in self.layers:
            inp = layer(inp)
            # print(f'input size: {inp.size()}')
        return inp



class Sequence_Generator(nn.Module):
    def __init__(self, embedding, decoder, fc_layer_dims, hidden_dim, target_length, output_dims, rnn_type='GRU'):
        """
        Entire vector -> str model (the 'encoder' in our problem setup)
        :param embedding: embedder class, where embedding(input) returns a vector
        :param decoder: (nn.Module)
        :param fc_layer_dims: dimensions of fully connected layers, EXCLUDING hidden dim
        :param target_length: the maximum length a sequence can be (max seq length)
        """
        super(Sequence_Generator, self).__init__()
        self.target_length = target_length
        self.output_dims = output_dims
        self.embedding = embedding
        self.decoder = decoder
        self.fc = FC_Encoder(fc_layer_dims)
        self.rnn_type = rnn_type
        # self.hidden_dim = hidden_dim
        #
        # param_size = sum([p.nelement() for p in self.parameters()]) # TODO - not sure where self.parameters is coming from
        # print('Total param size: {}'.format(param_size))

    def forward(self, input): # Input should be [x,y] value
        '''
        :param input: should be a [x,y] pair
        :return: output_tensor: tensor of word predictions in indexed form
        '''
        # get initial hidden and input
        print(f'input to seq generator: {input}')
        print(f'fully connected: {self.fc}')
        # pass through FCL
        init_hidden = self.fc(input)

        # if lstm, make sure to have both c and h as a tuple for the decoder
        if self.rnn_type == 'LSTM':
            init_hidden = (init_hidden, init_hidden)
        print(f'init hidden layer: {init_hidden}\nshape: {init_hidden.shape}')

        # initialize tensor of predictions (prediction for only one sentence)
        output_tensor = torch.zeros(self.target_length)
        output_tensor[0] = 1 # set first output to be <cls>

        # init decoder input and hidden
        decoder_input = None
        decoder_hidden = init_hidden

        print(f'starting rnn')
        # RNN loop
        for di in range(1, self.target_length):
            print(f'rnn loop {di}, before self.decoder')
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden) #TODO - handle LSTMs here too
            print(f'rnn loop {di}, after self.decoder')

            # get top index from softmax of previous layer
            topv, topi = decoder_output.topk(1)
            top_ix = topi.squeeze().detach()
            output_tensor[di] = top_ix

            decoder_input = top_ix
            if top_ix.item() == 2: # if end token
                break

        return output_tensor




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

