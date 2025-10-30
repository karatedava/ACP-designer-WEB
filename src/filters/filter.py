from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

class Filter(ABC):

    @abstractmethod
    def filter_sequences(df:pd.DataFrame) -> pd.DataFrame:

        """
        - Run filtering on df['sequences']
        RETURN: same dataframe with added collumn with filtering labels 
        """
    
    @abstractmethod
    def preprocess_sequences(sequences:np.ndarray) -> np.ndarray:

        """
        - convert sequences to correct format for filtering
        RETURN: array of sequences in correct format
        """