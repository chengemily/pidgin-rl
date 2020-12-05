import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embedding, rnn_type='GRU', nlayers=1, dropout=0.):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = embedding
        # self.dropout = nn.Dropout(dropout)

        # Define recurrent unit
        self.rnn_type = rnn_type
        rnn_cell = getattr(nn, rnn_type)  # get constructor from torch.nn
        self.rnn = rnn_cell(hidden_dim,  # 1st param - input size, 2nd param - hidden size
                            hidden_dim,
                            nlayers, dropout=dropout,  # ,
                            batch_first=True)  # if inputs are (batch_size, seq, feature)

        # Define params needed for output unit
        # Define linear and softmax units, assumes input of shape (batch, sentence_length, vector_length)
        self.out = nn.Linear(hidden_dim, output_dim) # 1st param - size of input, 2nd param - size of output
        self.softmax = nn.LogSoftmax(dim=0)  # dim=1 means take softmax across first dimension

    def forward(self, inp, h0):
        # if LSTM, change hidden state

        output = self.embedding(inp).view(-1, 1,
                                          self.hidden_dim)  # TODO - added a third dimension using unsqueeze, check later if errors
        # input(output.size())
        # output = torch.ones([32, 1, 300], device=torch.device("cuda:0")).float()
        # print(f'embedded input: {output.shape}')
        # output = self.dropout(output)

        # output = F.relu(output)

        # print(f'output shape: {output.size()}')

        # if isinstance(h0, tuple):
        # print(f'hidden shape: {h0[0].size()}')
        # else: print(f'hidden shape: {h0.size()}')

        # print(f'started rnn')
        output, hidden = self.rnn(output, h0)  # TODO - make sure h0 is a tuple if using LSTM, one val if gru
        # print(f'output of RNN unit: {output}')
        # print(f'hidden output of RNN unit: {hidden}')

        # pass output through fcl and softmax
        output = self.out(output) # take output[0]?
        # print(f'after self.out : {output}')
        output = self.softmax(output).float().squeeze()  # TODO - why output[0]?
        # print(f'after sotmax : {output}')
        return output, hidden


class FC_Encoder(nn.Module):
    def __init__(self, layer_dims=[]):
        super(FC_Encoder, self).__init__()
        layer_dims = [2] + layer_dims  # input is size 2 vector
        # print(f'fcl layer dims : {layer_dims}')
        self.layers = [nn.Linear(layer_dims[i], layer_dims[i + 1]).cuda() for i in range(len(layer_dims) - 1)]
        # print(self.layers)
        # TODO - add activation functions between layers?

    def forward(self, input):
        inp = input
        for layer in self.layers:
            inp = layer(inp)
            # print(f'input size: {inp.size()}')
        return inp


class Sequence_Generator(nn.Module):
    def __init__(self,
                 embedding,
                 decoder,
                 fc_layer_dims,
                 hidden_dim,
                 target_length,
                 output_dims,
                 batch_size,
                 vocab_size,
                 rnn_type='GRU',
                 device=torch.device("cuda:0")):
        """
        Entire vector -> str model (the 'encoder' in our problem setup)
        :param embedding: embedder class, where embedding(input) returns a vector
        :param decoder: (nn.Module)
        :param fc_layer_dims: dimensions of fully connected layers, EXCLUDING hidden dim
        :param target_length: the maximum length a sequence can be (max seq length)
        """
        super(Sequence_Generator, self).__init__()
        # define variable size inputs
        self.target_length = target_length
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        # self.hidden_dim = hidden_dim


        # store device for creating decoder_input and outputs tensor
        self.device = device

        # store all models needed
        self.embedding = embedding
        self.decoder = decoder
        self.fc = FC_Encoder(fc_layer_dims)
        self.rnn_type = rnn_type # needed to toggle LSTM/GRU model

        # param_size = sum([p.nelement() for p in self.parameters()]) # TODO - not sure where self.parameters is coming from
        # print('Total param size: {}'.format(param_size))

    def forward(self, inp): # Input should be [x,y] value
        '''
        :param input: should be a [x,y] pair
        :return: output_tensor: tensor of word predictions in indexed form
        '''
        batch_output = torch.zeros(self.batch_size, self.target_length, self.vocab_size, device=self.device)

        # [x,y] through FC to get initial hidden state
        init_hidden = self.fc(inp).unsqueeze(0)  # .to(device)  # hidden dim is (num_layers, batch, hidden_size)

        # print(f'initial hidden: {init_hidden}')
        # print(f'initial hidden size: {init_hidden.size()}')

        # make compatible with LSTM
        if rnn_type == 'LSTM':
            init_hidden = (init_hidden, init_hidden)
        # print(f'x after fcl, hidden batch : {init_hidden}')

        # init decoder hidden and input
        decoder_hidden = init_hidden
        decoder_input = torch.ones(self.batch_size, 1, dtype=torch.long,
                                   device=self.device)  # init starting tokens, long is the same as ints, which are needed for embedding layer

        # if isinstance(decoder_hidden, tuple):
        #     print(f'decoder hidden: size: {decoder_hidden[0].size()}')
        # else: print(f'decoder hidden size: {decoder_hidden.size()}')

        # run batch through rnn
        for di in range(1, self.target_length - 1):  # start with 1 to predict first non-cls word
            # print(f'rnn loop {di}, before self.decoder')
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # TODO - handle LSTMs here too
            # print(f'rnn loop {di}, after self.decoder')
            #
            # print(f'decoder output: {decoder_output}')
            # print(f'decoder output size: {decoder_output.size()}')
            # print(f'y: {y[:, di]}')
            # print(f'y size: {y[:, di].size()}')
            # input(decoder_output.size())

            # print(f'loss = {loss}')
            # get top index from softmax of previous layer
            topv, topi = decoder_output.topk(1)  # taking argmax
            topv.detach()  # detaching for safe measure
            decoder_input = topi.view(-1, 1).detach()

            # add decoder output to outputs tensor
            batch_output[:, di, :] = decoder_output

        # TODO - detach anything here?

        return batch_output




