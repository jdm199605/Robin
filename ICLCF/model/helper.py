import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


def pad_sequence(seq, max_len):
    for s in seq:
        num_zeros = max_len - len(s)
        s += [0] * num_zeros
    return seq

def encode(inputs, size, attr_num, model_name):
    model = BertModel.from_pretrained(model_name)
    _, max_len = inputs.shape
    with torch.no_grad():
        outputs = model(input_ids=inputs).last_hidden_state
    outputs = outputs.view(size, attr_num * max_len, 768).mean(dim=1)
    return outputs

def sequentialize_table(table, model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    columns = table.columns
    seq = []
    max_len = 0
    attr_num = len(columns)
    for i, t in table.iterrows():
        for j, column in enumerate(columns):
            if pd.isna(table.iloc[i][j]):
                input_ids = tokenizer.encode("[NULL]", add_special_tokens=True)
            else:
                input_ids = tokenizer.encode(str(table.iloc[i][j]), add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
            seq.append(input_ids)
    seq = pad_sequence(seq, max_len)
    seq = torch.tensor(seq).view(-1, max_len)
    return seq, attr_num, len(table)
    
    
def fill_template(t, attr, query):
    ##Instruction template: give a tuple, if its value on attribute __ is __, ..., respectively, then ...
    attr_num = len(t)
    templated_tuple = "Given a tuple,"
    start = 1
    for key in t.keys():
        if key == attr:
            continue
        if start:
            text = f"if its value on attribute {key} is {t[key]},"
        else:
            text = f"its value on attribute {key} is {t[key]},"
        templated_tuple += text
    
    templated_tuple += ',respectively,'
    
    if query:
        templated_tuple += 'then its value on attribute {attr} is'
    else:
        templated_tuple += 'then its value on attribute {attr} is {t[attr]}.'
    
    return templated_tuple()

def find_k_demonstration_examples(embeddings, target_embedding, k):
    cosine_similarities = F.cosine_similarity(embeddings, target_embedding.unsqueeze(0), dim=1)
    top_k_indices = torch.argsort(cosine_similarities, descending=True)[:k]
    return top_k_indices
    