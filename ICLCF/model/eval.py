from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
from helper import fill_template, sequentialize_table, find_k_demonstration_examples, sequentialize_table, encode
import torch
import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--k', type = int, default = 35)
args = parser.parse_args()

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

input_data = '../data/R1.json'
input_table = '../tables/R1/R1.csv'

with open(input_data, 'r') as file:
    data = json.load(file)
    
table = pd.read_csv(input_table, encoding='latin1', warn_bad_lines=True, error_bad_lines=False)
columns = table.columns

table_new = pd.DataFrame(columns = columns)

total = len(data)
correct = 0

for key in data.keys():
    t = data[key]['tuple']
    
    t = pd.DataFrame(t, index = [0])
    table_new = table_new.append(t, ignore_index = True)

table_seq, attr_num_1, size_1 = sequentialize_table(table, model_name)
table_new_seq, attr_num_2, size_2 = sequentialize_table(table_new, model_name) 

t_new_emb = encode(table_new, attr_num_1, size_1, model_name).mean(dim=0)
table_emb = encode(table, attr_num_1, size_1, model_name)

indices = find_k_demonstration_examples(t_new_emb, table_emb, args.k)

input_text = ""
for idx in indices:
    t_dict = table.iloc[idx].to_dict()
    attr = random.choice(columns)
    input_text += fill_template(t_dict, attr, 0)
    
for key in data.keys():
    ground_truth = data[key]['ground_truth']
    attr = data[key]['conflicted_attr'] 
    t = data[key]['tuple']
    candidate_set = data[key]['candidate_set']
    
    query_text = input_text + fill_template(t, attr, 1)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    beam_outputs = model.generate(
        input_ids,
        max_length=20,
        num_return_sequences=len(candidate_set),
        early_stopping=True,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    )
    
    predicted_words = []
    for beam_output in beam_outputs:
        predicted_word = tokenizer.decode(beam_output, skip_special_tokens=True)
        predicted_words.append(predicted_word)
    
    best_candidate = max(candidate_set, key=lambda candidate: predicted_words.count(candidate))
    
    if best_candidate == ground_truth:
        correct += 1

accuracy = correct / total

print (f'accuracy: {accuracy}')
    