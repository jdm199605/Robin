# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

class TableDataset(Dataset):
    def __init__(self, training_table, augmented_table, pos, neg):
        self.table = pd.read_csv(training_table, encoding = 'utf-8',  warn_bad_lines=True, error_bad_lines=False)  
        self.augmented_table =  pd.read_csv(augmented_table, encoding = 'utf-8',  \
                                            warn_bad_lines=True, error_bad_lines=False)
        self.pos = pos
        self.neg = neg
    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, index):
        target = torch.zeros(self.pos+self.neg)
        batch = []
        t = self.table.iloc[index]
        augmented_t = self.augmented_table[index * self.pos, (index+1) * self.pos]
        negative_t = np.random.choice(len(self.table), self.neg, replace=False)
        
        for i, t_ in enumerate(augmented_t):
            batch.append([t, t_])
            target[i] = 1
        
        for i, t_ in enumerate(negative_t):
            batch.append([t, t_])
            target[i] = 0
            
        return batch, target
        
        
