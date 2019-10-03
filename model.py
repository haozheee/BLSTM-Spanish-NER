import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.5, bidirectional=True, num_layers=2)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):  # x dim: batch_size x batch_max_len
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
