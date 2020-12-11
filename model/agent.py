import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Agent(nn.Module):
    def __init__(self, embedding, encoder_rnn, fc_encoder, decoder_rnn, fc_decoder, batch_size, vocab_size, target_length, device, end2end=False):
        """
        Wires encoder/decoder together as a comprehensive monolingual agent.
        :param embedding: (nn.Module) embedding model to share
        :param encoder_rnn: (nn.Module) LSTM or GRU RNN module converting vector to string
        :param fc_encoder: (nn.Module) FC network taking vector to 1st hidden layer of encoder_rnn
        :param decoder_rnn: (nn.Module) LSTM/GRU RNN converting string to vector
        :param fc_decoder: (nn.Module) FC network taking last hidden layer to 2x1 vector
        """
        super(Agent, self).__init__()
        self.end2end = end2end
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.target_length = target_length
        self.device = device
        
        # modules
        self.embedding = embedding if not self.end2end else self.make_embedding(embedding) # embedding to share
        self.encoder_rnn = encoder_rnn
        self.fc_encoder = fc_encoder
        self.decoder_rnn = decoder_rnn
        self.fc_decoder = fc_decoder

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

    def forward_encoder_normal(self, inp):
        assert not self.end2end

        batch_output = torch.zeros(self.batch_size, self.vocab_size, self.target_length, device=self.device)

        # [x,y] through FC to get initial hidden state
        init_hidden = self.fc_encoder(inp).unsqueeze(0)  # .to(device)  # hidden dim is (num_layers, batch, hidden_size)

        # make compatible with LSTM
        init_hidden = (init_hidden, init_hidden)

        # init encoder hidden and input
        encoder_hidden = init_hidden
        encoder_input = torch.ones(self.batch_size, 1, dtype=torch.long,
                                   device=self.device)  # init starting tokens, long is the same as ints, which are needed for embedding layer
        # add starting token to batch_output
        cls_matrix = torch.zeros(self.batch_size, self.vocab_size)
        cls_matrix[:, 1] = 1
        batch_output[:, :, 0] = cls_matrix

        # run batch through rnn
        for di in range(1, self.target_length - 1):  # start with 1 to predict first non-cls word
            # print(f'rnn loop {di}, before self.encoder')
            encoder_output, encoder_hidden = self.encoder_rnn(self.embedding(encoder_input),
                                                          encoder_hidden)  # TODO - handle LSTMs here too

            # get top index from softmax of previous layer
            topv, topi = encoder_output.topk(1)  # taking argmax, make sure dim is correct
            topv.detach()  # detaching for safe measure
            encoder_input = topi.view(-1, 1).detach()

            # add encoder output to outputs tensor
            batch_output[:, :, di] = encoder_output.squeeze()

        return batch_output

    def forward_encoder_end2end(self, inp):
        assert self.end2end

        # each word in the output sentence is a length-V vector
        batch_output = torch.zeros(self.batch_size, self.vocab_size, self.target_length, device=self.device)

        # [x,y] through FC to get initial hidden state
        init_hidden = self.fc_encoder(inp).unsqueeze(0)  # hidden dim is (num_layers, batch, hidden_size)

        # make compatible with LSTM
        init_hidden = (init_hidden, init_hidden)

        # init encoder hidden and input NOTE: just need to change inputs and outputs to be one-hot
        encoder_hidden = init_hidden
        encoder_input = torch.zeros(self.batch_size, self.vocab_size, dtype=torch.long,
                                    device=self.device)  # init starting tokens, long is the same as ints, which are needed for embedding layer
        encoder_input[:, 1] = 1  # one-hot vectors @ position 1.

        # add starting token to batch_output
        cls_matrix = torch.zeros(self.batch_size, self.vocab_size)
        cls_matrix[:, 1] = 1
        batch_output[:, :, 0] = cls_matrix

        # run batch through rnn
        for di in range(1, self.target_length - 1):  # start with 1 to predict first non-cls word
            # print(f'rnn loop {di}, before self.encoder')
            embedded = self.embedding(encoder_input.float())
            encoder_output, encoder_hidden = self.encoder_rnn(
                embedded.view(embedded.size()[0], 1, embedded.size()[1]),
                encoder_hidden
            )  # TODO - handle LSTMs here too
            # Decoduer output is a length V vector.
            # input("encoder output one word vector size: {}".format(encoder_output.size()))
            encoder_output = F.gumbel_softmax(encoder_output.squeeze(), hard=True)
            encoder_input = encoder_output

            # input("encoder output one word vector size: {}".format(encoder_output.size()))
            # add encoder output to outputs tensor
            batch_output[:, :, di] = F.gumbel_softmax(encoder_output.squeeze(), hard=True)

        return batch_output

    def forward_encoder(self, inp):
        """
        Forward pass for encoder module
        :param inp: [batchx2x1] vectors
        :return:
        """
        return self.forward_encoder_end2end(inp) if self.end2end else self.forward_encoder_normal(inp)


    def forward_decoder(self, input, x_lengths):
        """
        Forward pass for decoder module
        :param input: if self.end2end, input should be batch x V x seq_len. Otherwise batch x 1 x seq_len
        :return:
        """
        # print('input size: ', input.size())
        # print(self.embedding)
        input = self.embedding(torch.transpose(input, 1, 2)) # make batch x seq_len x embed size
        # print(self.decoder_rnn)
        outputs, hidden = self.decoder_rnn(input, x_lengths)
        # print(hidden[0].size())
        # print(outputs.size())
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[1]  # take the cell state
        if self.decoder_rnn.bidirectional:  # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        output_vec = self.fc_decoder(hidden)
        # print(output_vec.size())
        return output_vec
    