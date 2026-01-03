"""
FILTER BASED ON DISTRIBUTION (check for outliers)
- charge distribution
- hydrophobic moment
"""
from src.config import CHARGE_FILE, HYDROPHOBICITY_FILE
from src.filters.filter import Filter
import numpy as np
import math
import pandas as pd

from pathlib import Path
import json

class DistributionFilter(Filter):

    def __init__(self, og_data_path:Path, charge_file:str = CHARGE_FILE, hydrophobicity_file:str=HYDROPHOBICITY_FILE):

        super().__init__()

        self.charge_dict = json.load(open(charge_file,'r'))
        self.hydrophobicity_dict = json.load(open(hydrophobicity_file,'r'))

        df = pd.read_csv(og_data_path)
        self.base_charge_scores, self.base_hmom_scores = self.preprocess_sequences(df['sequence'])

    def filter_sequences(self, df:pd.DataFrame):

        """
        run filtering for charge / hydrophobic moment
        categories: OK / EDGE 
        RETURN: same pandas dataframe with added columns 'charge_d', 'hmom_d'
        """

        sequences = df['sequence'].to_list()
        charge_scores, hmom_scores = self.preprocess_sequences(sequences)

        ### IQR FILTERING ###

        ## CHARGE ##
        df['charge'] = charge_scores
        charge_lower, charge_upper = self._IQR_bounds_(self.base_charge_scores)

        df['d_charge'] = np.select(
            [df['charge'] < charge_lower, 
            df['charge'].between(charge_lower, charge_upper, inclusive='left'), 
            df['charge'] >= charge_upper],
            ['EDGE', 'OK', 'EDGE'],
            default='UKN'
        )

        ## HYDROPHOBIC MOMENT ##
        df['hmom'] = hmom_scores
        hmom_lower, hmom_upper = self._IQR_bounds_(self.base_hmom_scores)
        df['d_hmom'] = np.select(
            [df['hmom'] < hmom_lower, 
            df['hmom'].between(hmom_lower, hmom_upper, inclusive='left'), 
            df['hmom'] >= hmom_upper],
            ['EDGE', 'OK', 'EDGE'],
            default='UKN'
        )
        df = df.drop(columns=['hmom','charge'])
        return df

    def preprocess_sequences(self, sequences) -> tuple[np.ndarray, np.ndarray]:

        """
        - compute scores for charge / hmom
        RETURN: scores arrays
        """

        charges = np.zeros(len(sequences))
        hmoms = np.zeros(len(sequences))

        for i,seq in enumerate(sequences):
            charges[i] = self._compute_charge_(seq)
            hmoms[i] = self._compute_hmom_(seq)
        
        return charges, hmoms     
            
    def _compute_charge_(self, seq:str) -> int:

        """
        - compute charge for given sequence
        """
        
        charge = 0
        for aa in seq:
            charge += self.charge_dict[aa]
        
        return charge
    
    def _compute_hmom_(self, seq:str) -> float:

        """
        - compute hydrophobic moment for given sequence
        """

        x_sum, y_sum = 0.0, 0.0
        angle = 100 # angle between residues in a-helix (approximate)

        for i, aa in enumerate(seq):
            h = self.hydrophobicity_dict[aa]

            theta = math.radians(angle * i)
            x_sum += h * math.cos(theta)
            y_sum += h * math.sin(theta)
        
        hmom = math.sqrt(x_sum**2 + y_sum**2) / len(seq)
        return hmom

    def _IQR_bounds_(self, wanted_distribution) -> tuple[float, float]:

        """
        - return lower / upper bound for IQR of desired distribution 
        """

        Q1 = np.percentile(wanted_distribution, 15)
        Q3 = np.percentile(wanted_distribution, 85)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return lower, upper
