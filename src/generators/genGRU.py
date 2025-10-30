from src.generators.generator import Generator
from src.utils.utils_ml import Dataset, collate_fn_no_label, normalize_embeddings
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd

class GenGRU(Generator):

    def __init__(self, model_path, batch_size=8):

        super().__init__(model_path)

        self.batch_size = batch_size
    
    def generate_sequences(self, n_batches:int=10) -> pd.DataFrame:
        
        """
        - generate novel APC-MP like sequences
        - keep only unque sequences
        """

        print('generating new ACPs')
        sequences_tensor = self.model.sample(self.batch_size * n_batches, self.vocabulary)

        sequences_strings = []
        for i in tqdm(range(len(sequences_tensor))):
            seq = sequences_tensor[i]
            seq = self.vocabulary.tensor_to_seq(seq, debug=False)
            sequences_strings.append(str(seq))
        
        sequences_strings = np.unique(sequences_strings)

        df = pd.DataFrame({
            'sequence':sequences_strings,
            'label':['GEN'] * len(sequences_strings)
        })

        return df

    def get_embeddings(self, df:pd.DataFrame) -> np.ndarray:

        # construct dataloader
        dataloader = self._construct_dataloader_(df)

        embeddings = normalize_embeddings([self.model.get_embeddings(dataloader)])

        return embeddings
    
    def get_embedding_single(self, seq:str) -> np.ndarray:

        df = pd.DataFrame({
            'sequence':[seq],
            'label':['target']
        })

        dataloader = self._construct_dataloader_(df,bs=1)
        embedding = normalize_embeddings(self.model.get_embeddings(dataloader))

        return embedding

    def _construct_dataloader_(self, df:pd.DataFrame, bs:int=None):

        if not bs:
            bs = self.batch_size

        dataset = Dataset(df, self.vocabulary, with_label=False)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn_no_label, drop_last=False)

        return dataloader
