# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

from transformers import BertTokenizer, BertModel
from helper import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class TableDataset(Dataset):
    def __init__(self, training_table, augmented_table, pos, neg, model_name):
        self.table = pd.read_csv(training_table, encoding = 'utf-8',  warn_bad_lines=True, error_bad_lines=False)  
        self.augmented_table =  pd.read_csv(augmented_table, encoding = 'utf-8',  \
                                            warn_bad_lines=True, error_bad_lines=False)
    
        self.pos = pos
        self.neg = neg
        self.model_name = model_name
    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, index):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        labels = torch.tensor([1]*self.pos+[0]*self.neg)

        t = self.table.iloc[index]
        augmented_t = self.augmented_table.iloc[index * self.pos: (index+1) * self.pos]
        temp_table = self.table.drop(index)
        negative_index = np.random.choice(len(temp_table), self.neg, replace=True)
        negative_t = temp_table.iloc[negative_index]
        
        attr_num = len(self.table.columns)
        
        seq_t, seq_t_pos, seq_t_neg = [], [], []
        mask_t, mask_t_pos, mask_t_neg = [], [], []
        max_len = 0
        
        for i in range(attr_num):
            if pd.isna(t[i]):
                input_ids = tokenizer.encode("[NULL]", add_special_tokens=True)
                mask_t.append(0)
            else:
                input_ids = tokenizer.encode(str(t[i]), add_special_tokens=True)
                mask_t.append(1)
            max_len = max(max_len, len(input_ids))
            seq_t.append(input_ids)
        
        for i, t_ in augmented_t.iterrows():
            for j in range(attr_num):
                if pd.isna(t_[j]):
                    input_ids = tokenizer.encode("[NULL]", add_special_tokens=True)
                    mask_t_pos.append(0)
                else:
                    input_ids = tokenizer.encode(str(t_[j]), add_special_tokens=True)
                    mask_t_pos.append(1)
                max_len = max(max_len, len(input_ids))
                seq_t_pos.append(input_ids)
        
        for i, t_ in negative_t.iterrows():
            for j in range(attr_num):
                if pd.isna(t_[j]):
                    input_ids = tokenizer.encode("[NULL]", add_special_tokens=True)
                    mask_t_neg.append(0)
                else:
                    input_ids = tokenizer.encode(str(t_[j]), add_special_tokens=True)
                    mask_t_neg.append(1)
                max_len = max(max_len, len(input_ids))
                seq_t_neg.append(input_ids)
        
        seq_t = torch.tensor(pad_sequence(seq_t, max_len)).view(-1, attr_num*max_len)
        seq_t_pos = torch.tensor(pad_sequence(seq_t_pos, max_len)).view(-1, attr_num*max_len)
        seq_t_neg = torch.tensor(pad_sequence(seq_t_neg, max_len)).view(-1, attr_num*max_len)
        
        mask_t = torch.tensor(mask_t)
        mask_t_pos = torch.tensor(mask_t_pos).view(self.pos, -1)
        mask_t_neg = torch.tensor(mask_t_neg).view(self.neg, -1)
        
        seq_t = seq_t.repeat(self.pos+self.neg, 1)
        seq_t_ = torch.cat((seq_t_pos, seq_t_neg), dim = 0)
        
        mask_t = mask_t.repeat(self.pos+self.neg, 1)
        mask_t_ = torch.cat((mask_t_pos, mask_t_neg), dim = 0)
        
        #print (seq_t.shape)
        #print (seq_t_.shape)
        
        inputs = torch.cat((seq_t, seq_t_), dim = 1).view(-1, attr_num, max_len)
        masks = torch.cat((mask_t, mask_t_), dim = 1).view(-1, attr_num)
        
        return inputs, labels, masks
        
        
        
            
        
        
        
