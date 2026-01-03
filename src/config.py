from pathlib import Path

FOLDER_SIGNATURE = 'designer_run_XX' # XX will be replaced by run_id
#OUTPUT_DIR = Path('static/runs/GENERATE/run_420')
OUTPUT_DIR = Path('static/runs')
DIRNAME_TEMPLATE = 'acp_maker'

# path generative models
GEN_PATH = Path('src/models/generative')
GENGRU_PATH = Path('src/models/generative/peptide_generator_MP_FT.pt')

# path filters
CTT_PATH = Path('src/models/classificators/RFC_esm.pkl') # cytotoxicity filter
DIST_DATA_PATH = Path('src/data/acp_trainset.csv') # data for distribution-based filters
DROP_COLS = ['toxicity_prob']

# path mutation DBs
MUT_DB_PATH = Path('src/data/mut_DBs')

# other
DEVICE = 'cpu'
BATCH_SIZE = 12
VOCABULARY_FILE = 'src/aa_vocabulary.json'
CHARGE_FILE = 'src/charge.json'
HYDROPHOBICITY_FILE = 'src/hydrophobicity.json'