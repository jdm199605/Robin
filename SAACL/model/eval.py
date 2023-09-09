import torch
import torch.nn as nn
import argparse
import glob
from data import TableDataset
from generator import attribute_deletion, attribute_substitution, word_deletion, word_substitution, numerical_typos,\
    text_typos
import pandas as pd
import random
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from model import AICJNet
import torch.optim as optim
from data import TableDataset
from torch.utils.data import DataLoader
from helper import find_connected_components, HungarianAlgorithm, handle_tuple_pair



input_path = '../tables/R1/'
saved_model = '../checkpoints/ckp.pth'

model_name = 'bert-base-uncased'

model = AICJNet(model_name, 768, 8, 0, 256)
checkpoint = torch.load(saved_model)
model.load_state_dict(checkpoint['model_state_dict'])

table = pd.read_csv(input_path + 'R1.csv', encoding='latin1', warn_bad_lines=True, error_bad_lines=False)

with open(input_path+'pairwise.txt', 'r') as file:
    GT_pairwise = eval(file.read())

with open(input_path + 'integrable_sets.txt', 'r') as file:
    GT_sets = []
    line = file.readline()
    while line:
        GT_sets.append(eval(line))
        
size = len(table)

TP = 0
TN = 0
FP = 0
FN = 0

graphM = np.zeros((size,size))
graphD = dict()

for i in range(size):
    t1 = table.iloc[i]
    for j in range(i + 1, size):
        t2 = table.iloc[j]
        inputs, mask = handle_tuple_pair(t1, t2, model_name)
        integrability = torch.argmax(model(inputs, mask)[0], dim=1)[0]
        graphM[i, j] = 1
        graphM[j, i] = 1
        if integrability:
            if [i,j] in GT_pairwise:
                TP += 1
            else:
                FP += 1
        else:
            if [i,j] in GT_pairwise:
                FN += 1
            else:
                TN +=1
for node in range(size):
    graphD[node] = list(np.nonzero(graphM[node])[0])
    print (graphD[node])
    
sets = find_connected_components(graphD)

####bipartite graph construction####
len1 = len(sets)
len2 = len(GT_sets)

B = np.zeros(len1+len2, len1+len2)
for i in range(0, len1):
    node1 = set(sets[i])
    for i in range(len1, len2):
        node2 = set(sets[j])
        B[i][j] = len(node1.intersection(node2))/len(node1.union(node2))
    
Recall = TP / TP + FN
Precision = TP / TP + FP
similarity = HungarianAlgorithm(B)

print (f'Recall: {Recall}\n Precision: {Precision}\n similarity: {similarity}')


        
    




    