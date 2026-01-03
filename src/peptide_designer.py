"""
- Generating and filtering of peptides
- creating visualizations
    - toxicity distribution
    - interactive latent plot 
"""

from src.generators.genGRU import GenGRU
from src.generators.genProGen import GenProGen
from src.filters.f_cytotox import CytotoxicityFilter
from src.filters.f_distribution import DistributionFilter
import src.utils.utils_visualization as visualizations
from src.utils.gen_esm_embedds import Embedder

from typing import List
from pathlib import Path

class PeptideDesigner():

    def __init__(self, gen_path:Path, f_ctt=None, f_dist=None, device='cpu'):
        
        if gen_path.is_dir():
             print('directory')
             self.generator = GenProGen(gen_path)
        else:
            print('file')
            self.generator = GenGRU(gen_path)
        
        self.embedder = Embedder(device)

        self.filters = []

        if f_ctt:
            self.filters.append(CytotoxicityFilter(f_ctt, device))
        if f_dist:
            self.filters.append(DistributionFilter(f_dist))

    def run(self, n:int, output_dir:Path, drop_cols:List=[]):

        # generate sequences / initial dataframe [seq:label]
        df = self.generator.generate_sequences(n)
        df = df[df['sequence'].str.len() > 0]

        # run filtering 
        for filter in self.filters:
            df = filter.filter_sequences(df)
        
        # save general results 
        output_dir.mkdir(parents=True, exist_ok=True)
        visualizations.probability_distribution(df['toxicity_prob'], output_dir, col='red', name='toxicity')

        sequences = list(df['sequence'].to_numpy())
        embedds = self.embedder.get_embeddings(sequences)

        visualizations.latent_space_plot(embedds,df['toxicity_cat'], output_dir)
        df = df.drop(columns=drop_cols)
        df.to_csv(output_dir / 'generated_sequences.csv',index=False)
