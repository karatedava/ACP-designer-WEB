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

class PeptideMutator():

    def __init__(self, db:str = 'rnn_acp_mp'):

        """
        possible DBs are: [acp_mp / acp / amp]
        """

        # self.generator = GenGRU(gen_path)
        self.embedder = Embedder()

        df = None
        embeddings = None

        pth = MUT_DB_PATH / db
        
        if pth.exists():
            # load corresponding df
            df = pd.read_csv(pth / 'DB_filtered.csv', index_col=False)
            # load corresponding embeddings
            embeddings = np.load(pth / 'esm_embedds_filtered.npy', allow_pickle=True)
        
        dim = embeddings.shape[1]

        self.mutator = MutatorFaiss(df, embeddings, dim)
    
    def run(self, target_sequence:str, output_dir:Path):

        mut_df = self.sample_mutants(target_sequence)

        # save results + visualizations
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizations.probability_distribution(mut_df['toxicity_prob'], output_dir, col='red', name='toxicity')
        visualizations.probability_distribution(mut_df['d_charge'], output_dir, col='blue', name='charge')
        #visualizations.latent_space_plot(self.generator.get_embeddings(df),df['toxicity_cat'], output_dir)

        mut_df.to_csv(output_dir / 'mutants.csv',index=False)

    def sample_mutants(self, sequence_string:str) -> pd.DataFrame:

        # seq_embedded = self.generator.get_embedding_single(sequence_string)
        # seq_embedded = seq_embedded.reshape((1,400))

        seq_embedded = self.embedder.get_embeddings([sequence_string],bs=1)

        mut_df = self.mutator.retrive_mutants(og_sequence=seq_embedded,n=10).reset_index()

        return mut_df