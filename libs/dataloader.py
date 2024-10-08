import torch
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

    
class ChunkedNumpyDataset(Dataset):
    def __init__(self, input_paths, max_events=None, device=None):
        
        self.input_data = []
        
        for input_path in input_paths:
            if max_events:
                self.input_data.append(torch.from_numpy(np.load(input_path)[:max_events]).to(torch.float32))
            else:
                self.input_data.append(torch.from_numpy(np.load(input_path)).to(torch.float32))
        
        self.total_length = min(len(d) for d in self.input_data)
        assert all(len(d) == self.total_length for d in self.input_data), \
            "All input and target arrays must have the same length"
        
        if device:
            self.input_data = [data.pin_memory().to(device) for data in self.input_data]
    
    def __len__(self):
        return self.total_length 
    
    def __getitem__(self, idx):
        
        inputs = [data[idx] for data in self.input_data]
        return inputs


