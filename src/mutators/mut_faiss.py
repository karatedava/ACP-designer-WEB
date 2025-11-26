import faiss

import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

class MutatorFaiss:

    def __init__(self, df:pd.DataFrame, vdb:np.ndarray=[], embedd_dim:int=100):
        
        # init data

        self.df = df

        if len(vdb) > 0:
            self.vdb = vdb
        else:
            self.vdb = self._construct_vectorDB_(embedd_dim)
        self.vdb = normalize(self.vdb, norm='l2', axis=1).astype('float32')
        
        # init faiss indexer

        #self.indexer = faiss.IndexFlatL2(embedd_dim)
        self.indexer = faiss.IndexFlatIP(embedd_dim)
        self.indexer.add(self.vdb)
    
    def retrive_mutants(self, og_sequence:np.ndarray, n:int=3) -> pd.DataFrame:
        """
        - find n-closest peptides from og_sequence
        RETURN: dataframe with mutants + corresponding filtering statistics
        """

        distances, indices = self.indexer.search(og_sequence,k=n+1)
        distances, indices = distances[0,1:], indices[0,1:]

        mutant_df = self.df.iloc[indices].reset_index()
        mutant_df['distance'] = distances

        return mutant_df.sort_values(by='distance', ascending=True)

    def _construct_vectorDB_(self) -> np.ndarray:
        pass
