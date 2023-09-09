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
from helper import create_directory

parser = argparse.ArgumentParser()
parser.add_argument('--pos', type = int, default = 6, help = "Number of postive instances")
parser.add_argument('--neg', type = int, default = 20, help = "Number of negative instances")
parser.add_argument('--lr', type = float, default = 1e-6, help = "Learning rate in Adam optimizer")
parser.add_argument('--momentum', type = float, default = 0.9, help = "Momentum in Adam optimizer")
parser.add_argument('--epochs', type = int, default = 30, help = "Number of training epochs")
parser.add_argument('--load', type = int, default = 0, help = "Whether starting training from a checkpoint")
parser.add_argument('--heads', type = int, default = 8, help = "Number of heads in self-attention mechanism")
parser.add_argument('--epsilon', type = float, default = 0.01, help = "constraints of the perturbation vector")
parser.add_argument('--adv', type = int, default = 0, help = "whether using adversarial examples")
parser.add_argument('--cuda', type = int, default = 1, help = "whether using GPU to accelerate model training" )
parser.add_argument('--hidden_size', type = int, default = 256, help = "Hidden size of self-attention" )

args = parser.parse_args()

def finding_adversarial_examples(data, epsilon, gradient):
    sign_gradient = gradient.sign()
    perturbed_data = data + epsilon * sign_gradient
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


training_data_path = '../data/'
augmented_data_path = '../augmented_data/'
create_directory(augmented_data_path)
path_of_tables = glob.glob(training_data_path+'*')

for i, table_path in enumerate(path_of_tables):
    table_name = table_path.replace('\\','/').split('/')[-1]
    augmented_table_path = augmented_data_path + table_name
    
    table = pd.read_csv(table_path, encoding = 'utf-8', warn_bad_lines=True, error_bad_lines=False)
    
    columns = table.columns
    IsNum = dict()
    
    for i, column in enumerate(columns):
        if is_numeric_dtype(table[column]):
            IsNum[column] = 1
        else:
            IsNum[column] = 0
    
    augmented_table = pd.DataFrame(columns = columns)
    
    for idx in range(len(table)):
        for i in range(args.pos):
            t = table.iloc[idx]
            attr = random.choice(columns)
            while pd.isna(t[attr]):
                attr = random.choice(columns)
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
    
    augmented_table.to_csv(augmented_table_path, index = False)

device = torch.device("cuda:0" if args.cuda else "cpu")

model_name = 'bert-base-uncased'

model = AICJNet(model_name, 768, args.heads, device, args.hidden_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

checkpoint_path = '../checkpoints/ckp.pth'

if args.load:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
    
for epoch in range(args.epochs):
    #model.train()
    
    for idx, table_path in enumerate(path_of_tables):
        table_name = table_path.replace('\\','/').split('/')[-1]
        augmented_table_path = augmented_data_path + table_name
        
        train_dataset = TableDataset(table_path, augmented_table_path, args.pos, args.neg, model_name)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle = True)
        
        for inputs, labels, masks in train_loader:
            inputs, labels, masks = inputs.to(device), labels.to(device).squeeze(), masks.to(device)
            optimizer.zero_grad()
            outputs, last = model(inputs, masks)
            loss = criterion(outputs, labels)
            loss.backward()  
            optimizer.step()
            
            if args.adv:
                gradient = last.retain_grad().data
                perturbed_data = finding_adversarial_examples(last, args.epsilon, gradient)
                
                optimizer.zero_grad()
                perturbed_outputs = model(perturbed_data)
                adv_loss = criterion(perturbed_outputs, labels)
                total_loss = (loss + adv_loss)/2
                
                total_loss.backward()
                optimizer.step()
            
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), checkpoint_path)
        
    

    
    
                
                
                
            
        
    
    
    
    

