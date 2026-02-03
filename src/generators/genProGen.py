from src.generators.generator import Generator
from tqdm import tqdm

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import torch

class GenProGen(Generator):

    def __init__(self, model_path):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # List of non-standard/ambiguous tokens to ban
        non_standard = ['O', 'U', 'B', 'Z', 'J', '1', 'X']  # Add 'X' here if needed

        bad_tokens = []
        for aa in non_standard:
            token_id = self.tokenizer.convert_tokens_to_ids(aa)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                bad_tokens.append(token_id)

        self.logits_processor = LogitsProcessorList([RestrictTokensLogitsProcessor(bad_tokens)])

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,   # or torch.float16
            device_map="auto",            # automatically puts on GPU(s)
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = 0
            self.model.config.pad_token_id = 0
        
        self.model.config.num_hidden_layers = self.model.config.n_layer
        self.model.config.hidden_size       = self.model.config.n_embd
        self.model.config.num_attention_heads = self.model.config.n_head

        self.model.config.use_cache = True

    def generate_sequences(self, n:int=100):

        prompt_ids = torch.tensor([[1] for _ in range(n)], device=self.model.device)

        generated = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=30,
            repetition_penalty=1.15,
            eos_token_id=2,
            pad_token_id=0,
            num_return_sequences=1,     # already batched
            logits_processor=self.logits_processor
        )

        sequences = []
        for seq in generated:
            # Remove <1> and everything after first <2>
            protein = self.tokenizer.decode(seq, skip_special_tokens=True)[:-1]
            if '2' in protein:
                protein = protein[:protein.index('2')]
            if '1' in protein:
                continue
            if 'X' in protein:
                protein = protein[:protein.index('X')]
            sequences.append(protein)
        
        sequences = np.unique(sequences)

        df = pd.DataFrame({
            'sequence': sequences,
            'label':['GEN'] * len(sequences)
        })

        return df

    def get_embeddings(self):
        raise NotImplementedError

class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, bad_token_ids):
        self.bad_token_ids = bad_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bad_token_ids:
            scores[:, self.bad_token_ids] = -float('inf')
        return scores
