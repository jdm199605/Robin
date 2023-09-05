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

parser = argparse.ArgumentParser()
parser.add_argument('--pos', type = int, default = 6)
parser.add_argument('--neg', type = int, default = 20)
parser.add_argument('--lr', type = float, default = 1e-6)
parser.add_argument('--epochs', type = int, default = 30)
parser.add_argument('--load', type = int, default = 0)
parser.add_argument('--heads', type = int, default = 8)
parser.add_argument('--epsilon', type = float, default = 0.01)
parser.add_argument('--adv', type = int, default = 1)

args = parser.parse_args()

def finding_adversarial_examples(data, epsilon, gradient):
    sign_gradient = gradient.sign()
    perturbed_data = data + epsilon * sign_gradient
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


training_data_path = '../data/'
path_of_tables = glob.glob(training_data_path+'*')

for i, table_path in enumerate(path_of_tables):
    table_name = table_path.split('/')[-1]
    augmented_table_name = "augmented" + table_name
    
    table = pd.read_csv(table_path, encoding = 'utf-8', warn_bad_lines=True, error_bad_lines=False)
    
    columns = table.columns
    IsNum = dict()
    
    for i, column in enumerate(columns):
        if is_numeric_dtype(table[column]):
            IsNum[column] = 1
        else:
            IsNum[column] = 0
    
    augmented_table = pd.DataFrame(columns = column)
    
    for idx in range(len(table)):
        for i in range(len(args.pos)):
            t = table.iloc[idx]
            attr = np.random.randint(0, len(columns))
            while pd.isna(t[attr]):
                attr = np.random.randint(0, len(columns))
            if IsNum[attr]:
                op = np.random.randint(0, 2)
                if op == 0:
                    augmented_t = attribute_deletion(t, attr)
                else:
                    augmented_t = numerical_typos(t, attr)
            else:
                op = np.random.randint(0, 5)
                if op == 0:
                    augmented_t = attribute_deletion(t, attr)
                elif op == 1:
                    augmented_t = attribute_substitution(t, attr)
                elif op == 2:
                    augmented_t = word_substitution(t, attr)
                elif op == 3:
                    augmented_t = word_deletion(t, attr)
                else:
                    augmented_t = text_typos(t, attr)
            augmented_table = augmented_table.append(augmented_t)
    
    augmented_table.to_csv(training_data_path + augmented_table_name, index = False)

model_name = 'bert-base-uncased'

model = AICJNet(model_name, 768, args.heads)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.load:
    checkpoint_path = '../checkpoints/ckp.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
for epoch in range(args.epochs):
    model.train()
    
    for idx, table_path in enumerate(path_of_tables):
        table_name = table_path.split('/')[-1]
        augmented_table_name = "augmented" + table_name
        
        table = pd.read_csv(table_path, encoding = 'utf-8', warn_bad_lines=True, error_bad_lines=False)
        
        train_dataset = TableDataset(table_path, training_data_path + augmented_table_name, args.pos, args.neg)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle = True)
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  
            optimizer.step()
            
            if args.adv:
                gradient = inputs.grad.data
                perturbed_data = finding_adversarial_examples(inputs, args.epsilon, gradient)
                
                optimizer.zero_grad()
                perturbed_outputs = model(perturbed_data)
                adv_loss = criterion(perturbed_outputs, labels)
                total_loss = (loss + adv_loss)/2
                
                total_loss.backward()
                optimizer.step()
            
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), checkpoint_path)
        
    

    
    
                
                
                
            
        
    
    
    
    

