# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import os

def dfs(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    
    for neighbor in graph[node]:
        print (neighbor)
        if not visited[neighbor]:
            dfs(graph, neighbor, visited, component)

def find_connected_components(graph):
    num_nodes = len(graph)
    visited = [False] * num_nodes
    components = []
    
    for node in range(num_nodes):
        if not visited[node]:
            component = []
            dfs(graph, node, visited, component)
            components.append(component)
    
    return components

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def format_batch_texts(language_code, batch_texts):
    formed_batch = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formed_batch


def perform_translation(batch_texts, model, tokenizer, language):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts

def perform_backtranslation(batch_texts, first_model, first_model_tkn, second_model, second_model_tkn, \
                            first_language, second_language):
    intermediate_texts = perform_translation(batch_texts, first_model, first_model_tkn, first_language)
    #print (intermediate_texts)
    final_texts = perform_translation(intermediate_texts, second_model, second_model_tkn, second_language)
    
    return final_texts

class HungarianAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.V = len(graph)
        self.matchR = [-1] * self.V
        
    def bipartite_matching(self, u, seen):
        for v in range(self.V):
            if self.graph[u][v] and not seen[v]:
                seen[v] = True
                
                if self.matchR[v] == -1 or self.bipartite_matching(self.matchR[v], seen):
                    self.matchR[v] = u
                    return True
        
        return False
    
    def max_bipartite_matching(self):
        result = 0
        for u in range(self.V):
            seen = [False] * self.V
            if self.bipartite_matching(u, seen):
                result += 1
        return result