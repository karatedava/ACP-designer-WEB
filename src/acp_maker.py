from src.generators.genGRU import GenGRU
from src.filters.f_cytotox import CytotoxicityFilter
from src.filters.f_distribution import DistributionFilter
from src.mutators.mut_faiss import MutatorFaiss
import src.utils.utils_visualization as visualizations

from typing import List
from pathlib import Path

class ACPmaker():

    def __init__(self, gen_path:Path, f_ctt=None, f_dist=None, device='cpu'):

        print(device)

        self.generator = GenGRU(gen_path)
    
        self.filters = []

        if f_ctt:
            self.filters.append(CytotoxicityFilter(f_ctt, device))
        if f_dist:
            self.filters.append(DistributionFilter(f_dist))

        self.mutator = None

    def run_pipeline(self, nbatches:int, output_dir:Path, drop_cols:List=[], to_mutate:str=None):

        # generate sequences / initial dataframe [seq:label]
        df = self.generator.generate_sequences(nbatches)
        df = df[df['sequence'].str.len() > 0]
        mut_df = None

        # run filtering 
        for filter in self.filters:
            df = filter.filter_sequences(df)

        # prepare mutator
        if to_mutate:
            print('\nSTARTING MUTANT SEARCH\n')
            embedded_sequences = self.generator.get_embeddings(df)
            self.mutator = MutatorFaiss(df.drop(columns=drop_cols), embedded_sequences, embedded_sequences.shape[1])
            mut_df = self.sample_mutants(to_mutate)

        # save general results 
        output_dir.mkdir(parents=True, exist_ok=True)
        visualizations.probability_distribution(df['toxicity_prob'], output_dir)
        visualizations.latent_space_plot(self.generator.get_embeddings(df),df['toxicity_cat'], output_dir)
        df = df.drop(columns=drop_cols)
        df.to_csv(output_dir / 'results_all.csv',index=False)
        if mut_df is not None:
            mut_df.to_csv(output_dir / 'mutants.csv',index=False)


    def sample_mutants(self, base_sequence:str):

        """
        - embedd base sequence
        - run knn in generated_sequences embedding space
        - return k nearest sequences
        """

        if self.mutator is None:
            print('mutator is not initialized !')
            return None

        # embedd the target sequence
        seq_embedded = self.generator.get_embedding_single(base_sequence)
        seq_embedded = seq_embedded.reshape((1,400))

        df = self.mutator.retrive_mutants(og_sequence=seq_embedded,n=3).reset_index()

        return df

        

        
