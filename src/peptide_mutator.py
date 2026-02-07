"""
Sampling mutants from large precomputed DB
"""

#from src.generators.genGRU import GenGRU
from src.utils.gen_esm_embedds import Embedder
from src.mutators.mut_faiss import MutatorFaiss
import src.utils.utils_visualization as visualizations

from src.config import MUT_DB_PATH

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

class PeptideMutator():

    def __init__(self, db: Union[str, pd.DataFrame] = 'rnn_acp_mp'):

        """
        possible DBs are: [acp_mp / acp / amp]
        """

        # self.generator = GenGRU(gen_path)
        self.embedder = Embedder()

        df = None
        embeddings = None

        if isinstance(db, pd.DataFrame):
            df = db.copy()
            embeddings = self.embedder.get_embeddings(df['sequence'].to_list())
        elif isinstance(db, str):
            pth = MUT_DB_PATH / db
            
            if pth.exists() and (pth / 'DB_filtered.csv').exists():
                try:
                    df = pd.read_csv(
                        pth / 'DB_filtered.csv',
                        index_col=False,
                    )

                    embeddings = np.load(pth / 'esm_embedds_filtered.npy', allow_pickle=True)

                except Exception as e:
                    raise ValueError(f"Failed to load CSV from {pth}/DB_filtered.csv: {e}")
            else:
                raise FileNotFoundError(
                    f"Database folder not found or missing DB_filtered.csv: {pth}"
                )
        
        else:
            raise TypeError(
                f"db must be either str (database name) or pd.DataFrame. "
                f"Got: {type(db).__name__}"
            )
        
        dim = embeddings.shape[1]

        self.mutator = MutatorFaiss(df, embeddings, dim)
    
    def run(self, target_sequence:str, output_dir:Path):

        mut_df = self.sample_mutants(target_sequence)

        # save results + visualizations
        output_dir.mkdir(parents=True, exist_ok=True)

        if 'toxicity_prob' in mut_df.columns:
            visualizations.probability_distribution(mut_df['toxicity_prob'], output_dir, col='red', name='toxicity')
        if 'd_charge' in mut_df.columns:
            visualizations.probability_distribution(mut_df['d_charge'], output_dir, col='blue', name='charge')

        mut_df.to_csv(output_dir / 'mutants.csv',index=False)

    def sample_mutants(self, sequence_string:str) -> pd.DataFrame:

        # seq_embedded = self.generator.get_embedding_single(sequence_string)
        # seq_embedded = seq_embedded.reshape((1,400))

        seq_embedded = self.embedder.get_embeddings([sequence_string],bs=1)

        mut_df = self.mutator.retrive_mutants(og_sequence=seq_embedded,n=10).reset_index()

        return mut_df