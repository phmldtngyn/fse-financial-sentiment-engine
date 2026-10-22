import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class FSEDataset(Dataset):
    """PyTorch Dataset cho SFT gold standard.

    Input parquet phải có 3 cột: text (str), score (float ∈ [-1,1]), tier (int ∈ {1,2,3}).
    """

    def __init__(self, parquet_path, tokenizer_name="VinAI/phobert-base-v2", max_len=256):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(
            r["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score": torch.tensor(float(r["score"]), dtype=torch.float),
            "tier": torch.tensor(int(r["tier"]) - 1, dtype=torch.long),  # 0-indexed
        }
class FSEDataset3Class(Dataset):
    """Dataset cho 3-class classification.

    Parquet phải có: text (str), label (int 0/1/2), score (float, optional).
    """

    def __init__(self, parquet_path, tokenizer_name="VinAI/phobert-base-v2", max_len=256):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(
            r["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
        }