import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, model_name):
        super(Encoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
    def forward(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors = "pt", padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

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
    