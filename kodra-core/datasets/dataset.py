import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from tokenizer.char_tokenizer import CharTokenizer

class CodeDataset(Dataset):
    """
    PyTorch Dataset for Causal Language Modeling in Kodra AI.
    Generates input sequences x and target sequences y shifted by 1 position.
    """
    def __init__(self, text: str, tokenizer: CharTokenizer, context_length: int = 256):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.tokens) - self.context_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.tokens) <= self.context_length:
            # Pad if shorter than context length
            padding = torch.zeros(self.context_length + 1 - len(self.tokens), dtype=torch.long)
            chunk = torch.cat([self.tokens, padding])
        else:
            chunk = self.tokens[idx : idx + self.context_length + 1]

        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def create_dataloader(text: str, tokenizer: CharTokenizer, context_length: int = 256, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    dataset = CodeDataset(text, tokenizer, context_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
