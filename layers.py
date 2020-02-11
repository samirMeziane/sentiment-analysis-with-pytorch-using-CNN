import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size=12000, embedding_dim=300, n_filters=100, filter_sizes=[4,3,5], output_dim=1, 
                 dropout=0.5,padding_idx=12345):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (fs, embedding_dim)) for fs in filter_sizes])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, text):
        x = self.embedding(text)

        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        x = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]

        x = t.cat(x, 1)

        x = self.dropout(x)

        logit = self.fc(x)
          
        return logit