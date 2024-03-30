import torch
import torch.nn as nn

# set random seed for reproducibility
torch.manual_seed(0)


class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):

        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.long()

        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        final_state = lstm_out[:, -1, :]
        out = self.fc(final_state)
        out = self.sigmoid(out)

        return out