from pathlib import Path

FOLDER_SIGNATURE = 'designer_run_XX' # XX will be replaced by run_id
OUTPUT_DIR = Path('static/runs')
DIRNAME_TEMPLATE = 'acp_maker'

# path generative models
GENGRU_PATH = Path('src/models/generative/peptide_generator_MP_FT.pt')

# path filters
CTT_PATH = Path('src/models/classificators/RFC_esm.pkl')
DIST_DATA_PATH = Path('src/data/acp_trainset.csv')
DROP_COLS = ['toxicity_prob']

DEVICE = 'cpu'
BATCH_SIZE = 12
VOCABULARY_FILE = 'src/aa_vocabulary.json'
CHARGE_FILE = 'src/charge.json'
HYDROPHOBICITY_FILE = 'src/hydrophobicity.json'