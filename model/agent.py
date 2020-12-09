class Agent(nn.Module):
    def __init__(self, embedding, encoder_rnn, fc_encoder, decoder_rnn, fc_decoder):
        """
        Wires encoder/decoder together as a comprehensive monolingual agent.
        :param embedding: (nn.Module) embedding model to share
        :param encoder_rnn: (nn.Module) LSTM or GRU RNN module converting vector to string
        :param fc_encoder: (nn.Module) FC network taking vector to 1st hidden layer of encoder_rnn
        :param decoder_rnn: (nn.Module) LSTM/GRU RNN converting string to vector
        :param fc_decoder: (nn.Module) FC network taking last hidden layer to 2x1 vector
        """
        super(Agent, self).__init__()
        
        # modules
        self.embedding = embedding # embedding to share
        self.encoder_rnn = encoder_rnn
        self.fc_encoder = fc_encoder
        self.decoder_rnn = decoder_rnn
        self.fc_encoder = fc_decoder
        
    
