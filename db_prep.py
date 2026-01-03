from src.filters.f_cytotox import CytotoxicityFilter
from src.filters.f_distribution import DistributionFilter
from src.utils.gen_esm_embedds import Embedder

from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FOLDER = Path('src/data/mut_DBs/progen2-acp')

CTT_PATH = Path('src/models/classificators/RFC_esm.pkl')
DIST_DATA_PATH = Path('src/data/acp_trainset.csv')
OG = pd.read_csv('src/data/known_acps/acp_mp.csv',index_col=False)

filters = []
filters.append(CytotoxicityFilter(CTT_PATH, 'mps'))
filters.append(DistributionFilter(DIST_DATA_PATH))

print('\n removing existing ACPs ... \n')

### remove already existing acps ###
DB = pd.read_csv(INPUT_FOLDER / 'DB_raw.csv',index_col=False)
DB_filtered = DB[~DB['sequence'].isin(set(OG['sequence']))]
DB_filtered.to_csv(INPUT_FOLDER / 'DB_filtered.csv', index=False)

print('\n filtering ... \n')

for filter in filters:
    DB_filtered = filter.filter_sequences(DB_filtered)

DB_filtered.to_csv(INPUT_FOLDER / 'DB.csv',index=False)

sequences = list(DB_filtered['sequence'].to_numpy())

print('\n generating ESM embeddings \n')

embedder = Embedder('mps')
embeddings = embedder.get_embeddings(sequences)
np.save(INPUT_FOLDER / 'esm_embedds_filtered.npy',embeddings)