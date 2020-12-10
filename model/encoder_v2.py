import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, emsize, rnn_type='GRU', nlayers=1, dropout=0.):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.emsize = emsize

        # Define recurrent unit
        self.rnn_type = rnn_type
        rnn_cell = getattr(nn, rnn_type)  # get constructor from torch.nn
        self.rnn = rnn_cell(emsize,  # 1st param - input size, 2nd param - hidden size
                            hidden_dim,
                            nlayers, dropout=dropout,  # ,
                            batch_first=True)  # if inputs are (batch_size, seq, feature)

        # Define params needed for output unit
        # Define linear and softmax units, assumes input of shape (batch, sentence_length, vector_length)
        self.out = nn.Linear(hidden_dim, output_dim) # 1st param - size of input, 2nd param - size of output
        # self.softmax = nn.LogSoftmax(dim=0)  # dim=1 means take softmax across first dimension

    def forward(self, inp, h0):
        """

        :param inp: embedded word
        :param h0:
        :return:
        """
        output, hidden = self.rnn(inp, h0)  # TODO - make sure h0 is a tuple if using LSTM, one val if gru

        # pass output through fcl and softmax
        output = self.out(output) # take output[0]?

        return output, hidden


class FC_Encoder(nn.Module):
    def __init__(self, layer_dims=[]):
        super(FC_Encoder, self).__init__()
        layer_dims = [2] + layer_dims  # input is size 2 vector
        # print(f'fcl layer dims : {layer_dims}')
        self.layers = [nn.Linear(layer_dims[i], layer_dims[i + 1]).cuda() for i in range(len(layer_dims) - 1)]
        # print(self.layers)

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

        # store device for creating decoder_input and outputs tensor
        self.device = device

        # store all models needed
        self.embedding = embedding
        self.decoder = decoder
        self.fc = FC_Encoder(fc_layer_dims)
        self.rnn_type = rnn_type # needed to toggle LSTM/GRU model


    def forward(self, inp): # Input should be [x,y] value
        '''
        :param input: should be a [x,y] pair
        :return: output_tensor: tensor of word predictions in indexed form
        '''
        batch_output = torch.zeros(self.batch_size, self.vocab_size, self.target_length, device=self.device)

        # [x,y] through FC to get initial hidden state
        init_hidden = self.fc(inp).unsqueeze(0)  # .to(device)  # hidden dim is (num_layers, batch, hidden_size)

        # make compatible with LSTM
        if self.rnn_type == 'LSTM':
            init_hidden = (init_hidden, init_hidden)

        # init decoder hidden and input
        decoder_hidden = init_hidden
        decoder_input = torch.ones(self.batch_size, 1, dtype=torch.long,
                                   device=self.device)  # init starting tokens, long is the same as ints, which are needed for embedding layer
        # add starting token to batch_output
        cls_matrix = torch.zeros(self.batch_size, self.vocab_size)
        cls_matrix[:,1] = 1
        batch_output[:,:,0] = cls_matrix

        # run batch through rnn
        for di in range(1, self.target_length - 1):  # start with 1 to predict first non-cls word
            # print(f'rnn loop {di}, before self.decoder')
            decoder_output, decoder_hidden = self.decoder(self.embedding(decoder_input), decoder_hidden)  # TODO - handle LSTMs here too

            # get top index from softmax of previous layer
            topv, topi = decoder_output.topk(1)  # taking argmax, make sure dim is correct
            topv.detach()  # detaching for safe measure
            decoder_input = topi.view(-1, 1).detach()

            # add decoder output to outputs tensor
            batch_output[:, :, di] = decoder_output.squeeze()

        return batch_output




