import torch
import numpy as np
from transformers import EsmModel, AutoTokenizer
from tqdm import tqdm

class Embedder():

    def __init__(self, device='cpu', modelname='esm2_t6_8M_UR50D'):

        ### initialize 
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(f'facebook/{modelname}')
        model = EsmModel.from_pretrained(f'facebook/{modelname}',dtype=torch.float16)
        model = model.to(device)
        self.model = model.half()
    
    @torch.no_grad
    def get_embeddings(self, seqs:np.array, maxlen:int=50, bs:int = 0) -> np.array:
        
        if bs == 0:
            if len(seqs) >= 1000:
                bs=64
            if len(seqs) >= 10000:
                bs=128
            else:
                bs = 4

        embeddings = []

        for i in tqdm(range(0, len(seqs), bs)):

            batch = seqs[i:i+bs]

                    # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=maxlen,
                truncation=True,
                padding=True
            ).to(self.device)

            outputs = self.model(**inputs)
            batch_embedds = outputs.last_hidden_state.cpu().numpy()
            batch_embedds = batch_embedds[:, 1:-1, :] # removing <cls> and <eos> tokens
            batch_embedds_mean = batch_embedds.mean(axis=1)

            embeddings.append(batch_embedds_mean)

        embeddings = np.concatenate(embeddings,axis=0)

        return embeddings
