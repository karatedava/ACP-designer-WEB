"""
- Generating and filtering of peptides
- creating visualizations
    - toxicity distribution
    - interactive latent plot 
"""

from src.generators.genGRU import GenGRU
from src.filters.f_cytotox import CytotoxicityFilter
from src.filters.f_distribution import DistributionFilter
import src.utils.utils_visualization as visualizations

from typing import List
from pathlib import Path

class PeptideDesigner():

    def __init__(self, gen_path:Path, f_ctt=None, f_dist=None, device='cpu'):
        
        self.generator = GenGRU(gen_path)

        self.filters = []

        if f_ctt:
            self.filters.append(CytotoxicityFilter(f_ctt, device))
        if f_dist:
            self.filters.append(DistributionFilter(f_dist))

    def run(self, nbatches:int, output_dir:Path, drop_cols:List=[]):

        # generate sequences / initial dataframe [seq:label]
        df = self.generator.generate_sequences(nbatches)
        df = df[df['sequence'].str.len() > 0]

        # run filtering 
        for filter in self.filters:
            df = filter.filter_sequences(df)
        
        # save general results 
        output_dir.mkdir(parents=True, exist_ok=True)
        visualizations.probability_distribution(df['toxicity_prob'], output_dir, col='red', name='toxicity')
        visualizations.latent_space_plot(self.generator.get_embeddings(df),df['toxicity_cat'], output_dir)
        df = df.drop(columns=drop_cols)
        df.to_csv(output_dir / 'generated_sequences.csv',index=False)
