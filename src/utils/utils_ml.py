import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Vocabulary():
    def __init__(self, vocabulary):
        if "<pad>" not in vocabulary:
            vocabulary = { character: value+1 for character, value in vocabulary.items() }
            vocabulary['<pad>'] = 0    
        if "<start>" not in vocabulary:
            vocabulary["<start>"] = len(vocabulary)
        if "<end>" not in vocabulary:
            vocabulary["<end>"] = len(vocabulary)
        self.inv_vocabulary = {v: k for k, v in vocabulary.items()}
        self.vocabulary = vocabulary
    
    def letter_to_index(self, letter):
        return self.vocabulary[letter]

    def index_to_letter(self, index):
        return self.inv_vocabulary[index]
    
    def tensor_to_seq(self, tensor, debug=False):
        seq = ''
        for element in tensor:
            new_char = self.inv_vocabulary[element.item()]
            if not debug:
                if new_char == "<pad>" or new_char == "<start>" or new_char == "<end>":
                    new_char = ''
            seq = seq + new_char
        return seq
    
    def __len__(self):
        return len(self.vocabulary)
    
    def seq_to_tensor(self, seq):

        """
        - Converts sequence to numeric tensor according the vocabulary
        - starts with <start>, ends with <end>
        """

        tensor = torch.zeros(len(seq)+2, dtype=torch.long)
        tensor[0] = torch.tensor(self.vocabulary["<start>"])
        for i, letter in enumerate(seq):
            tensor[i+1] = torch.tensor(self.vocabulary[letter])
        tensor[-1] = torch.tensor(self.vocabulary["<end>"])
        return tensor
    
    @classmethod
    def get_vocabulary_from_sequences(cls, seq):
        # returns vocabolary class initialized with the vocabulary of the give X_seq
        count_vect = CountVectorizer(lowercase = False, analyzer = "char")
        count_vect.fit(seq)
        char_voc = count_vect.vocabulary_
        char_voc_ = {}
        for k,v in char_voc.items():
            char_voc_[k] = v + 1
        char_voc_['<pad>'] = 0
        return Vocabulary(char_voc_)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, vocabulary : Vocabulary, with_label=True):
        self.vocabulary = vocabulary
        self.dataset = dataset
        self.category = with_label
    
    def __getitem__(self, i):
        seq_str = self.dataset.iloc[i]["sequence"]
        tensor = self.vocabulary.seq_to_tensor(seq_str)
        if self.category:
            category_tensor = torch.tensor(self.dataset.iloc[i]["label"], dtype=torch.long)
            return tensor, category_tensor
        else:
            return tensor

    def __len__(self):
        return len(self.dataset)

def collate_fn(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    arr_seq = [element[0] for element in arr] # array of all sequences 
    cat_seq = [element[1] for element in arr] # array of corresponding labels

    # combine sequences into a single tensor along with metadata (batch size) and tracks the lenght of each sequence in the batch
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr_seq, enforce_sorted=False)

    # convert to padded tensor format, adding zero padding to shorter sequences, 
    # produces zero padded sequences, also contains information about their unpadded lenght
    padded_seq = torch.nn.utils.rnn.pad_packed_sequence(packed_seq,batch_first=True)
    return padded_seq, torch.tensor(cat_seq,dtype=torch.float32)

def collate_fn_no_label(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr, enforce_sorted=False)
    padded_seq = torch.nn.utils.rnn.pad_packed_sequence(packed_seq,batch_first=True)

    return padded_seq

def normalize_embeddings(embedd_arrs) -> np.ndarray:

    """
    L2 normalization of embeddings
    - return normalized embeddigns with corresponding labels
    """

    embedding_matrix = np.concatenate(embedd_arrs)
    normed_embedd_matrix = (embedding_matrix - embedding_matrix.mean(0)) / embedding_matrix.std(0)

    return normed_embedd_matrix