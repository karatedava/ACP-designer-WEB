import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np

class GenRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size
        )
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.output_layer = nn.Linear(
            in_features=hidden_size,
            out_features=input_size
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, sequences, lenghts, hidden=None):
        output = self.embedding(sequences)

        output = nn.utils.rnn.pack_padded_sequence(output, lenghts, batch_first=True, enforce_sorted=False)

        all_outputs, hidden = self.rnn(output,hidden)

        all_outputs, _ = nn.utils.rnn.pad_packed_sequence(all_outputs,batch_first=True)

        output = self.output_layer(all_outputs)

        return output, hidden
    
    def entropy(self, sequences, lenghts):

        output, _ = self.forward(sequences,lenghts)
        tensor_expected = torch.zeros_like(sequences)
        tensor_expected[:,:-1] = sequences[:,1:]

        entropies = torch.zeros(sequences.size()[0], device=sequences.device)

        for i, lenght in enumerate(lenghts):
            entropies[i] = F.cross_entropy(output[i,:lenght-1],tensor_expected[i,:lenght-1],reduction='sum',ignore_index=0)
        return entropies
    
    @torch.no_grad()
    def get_embeddings(self,dataloader,hidden=None):

        embeddings = []

        for i_batch, sample_batched in enumerate(dataloader):
            sequences = sample_batched[0].to('cpu')
            lenghts = sample_batched[1].to('cpu')

            output = self.embedding(sequences)

            output = nn.utils.rnn.pack_padded_sequence(output, lenghts, batch_first=True, enforce_sorted=False)

            all_outputs, hidden = self.rnn(output,hidden)

            all_outputs, _ = nn.utils.rnn.pad_packed_sequence(all_outputs,batch_first=True)

            embeddings.append(hidden[-1].numpy())

        embeddings = np.concatenate(embeddings)
        return embeddings
    
    @torch.no_grad()
    def sample(self, batch_size, vocabulary, max_len=40):

        start_token = torch.zeros(batch_size, device='cpu', dtype=torch.long)
        start_token[:] = vocabulary.vocabulary['<start>']
        start_token = start_token.unsqueeze(dim=1)

        end_token = torch.zeros_like(start_token)
        end_token[:] = vocabulary.vocabulary['<end>']

        hidden = None

        lengths = torch.ones(batch_size, dtype=torch.long, device='cpu')

        sequences = []
        sequences.append(start_token)

        s = torch.zeros(batch_size,device='cpu')

        unfinished = torch.ones_like(start_token)

        for _ in range(max_len):

            prob, hidden = self.forward(sequences[-1],lengths,hidden)

            prob = self.softmax(prob)
            sampled_characters = torch.multinomial(prob.view(-1, prob.size()[-1]),1)
            sampled_characters = sampled_characters * unfinished

            sequences.append(sampled_characters)

            unfinished[(sampled_characters == end_token)] = 0

            if unfinished.sum() == 0:
                break
        
        sequences = torch.cat(sequences,1)
        return sequences