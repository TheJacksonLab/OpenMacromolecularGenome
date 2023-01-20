from torch import nn


class Decoder(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size,
                 out_dimension, bidirectional=True):
        """
        Through Decoder
        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dimension = out_dimension
        self.bidirectional = bidirectional

        # simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.out_dimension) if bidirectional else
            nn.Linear(self.hidden_size, self.out_dimension)
        )

    def init_hidden(self, z):
        z_to_rnn = z.unsqueeze(0)
        if self.bidirectional:
            num_layers = self.num_layers * 2
        else:
            num_layers = self.num_layers

        return z_to_rnn.repeat(num_layers, 1, 1)

    def forward(self, x, hidden):
        """
        A forward pass throught the entire model.
        """

        # Decode
        output, hidden = self.decode_RNN(x, hidden)
        output = self.decode_FC(output)  # fully connected layer

        return output, hidden
