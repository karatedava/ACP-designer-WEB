from abc import ABC, abstractmethod
from src.config import VOCABULARY_FILE
from pathlib import Path
import src.utils.utils_ml as ml

import pandas as pd
import numpy as np
import torch
import json

class Generator(ABC):

    def __init__(self, model_path:Path):
        super().__init__()

        # load vocabulary
        self.vocabulary = self._load_vocabulary_()
        # load model
        self.model = torch.load(model_path,weights_only=False)
    
    @abstractmethod
    def generate_sequences(self) -> pd.DataFrame:

        """
        Generate protein sequences
        RETURN: dataframe [sequence: label]of generated sequences
        """
    
    @abstractmethod
    def get_embeddings(self) -> np.ndarray:

        """
        Get embeddings of sequences
        - latent represetation used for visualization purposes
        RETURN: normalized embedding matrix
        """
    
    def _load_vocabulary_(self):

        with open(VOCABULARY_FILE,'r') as f:
            return ml.Vocabulary(json.load(f))