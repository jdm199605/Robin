# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import glob
import numpy as np
import typo
from helper import perform_backtranslation
from transformers import MarianMTModel, MarianTokenizer, pipeline
import nltk
from nltk.corpus import wordnet
import random


###Attribute Level Data Augmentation

first_model_name = 'Helsinki-NLP/opus-mt-en-fr'
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
first_model = MarianMTModel.from_pretrained(first_model_name)

second_model_name = 'Helsinki-NLP/opus-mt-fr-en'
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
second_model = MarianMTModel.from_pretrained(second_model_name)

IntTypoNames = ['digit_swap','missing_digit', 'extra_digit', 'nearby_digit', 'similar_digit', 'repeated_digit', 'unidigit']
StrTypoNames = ['char_swap', 'missing_char', 'extra_char', 'nearby_char', 'similar_char', 'skipped_space', 'random_space'\
                'repeated_char', 'unichar']
    
def attribute_deletion(t, attr):
    t[attr] = np.nan
    return t
    
def attribute_substitution(t, attr):
    original_text = [t[attr]]
    backtranslated_text = perform_backtranslation(original_text, first_model, first_model_tkn, second_model, \
                                              second_model_tkn, "fr", "en")[0]
    t[attr] = backtranslated_text
    return t

def word_deletion(t, attr):
    text = t[attr]
    words = nltk.word_tokenize(text)
    if len(words) > 1:
        word_to_remove = random.choice(words)
        words.remove(word_to_remove)
        text_without_word = ' '.join(words)
        t[attr] = text_without_word
    return t

def word_substitution(t, attr):
    text = t[attr]
    words = nltk.word_tokenize(text)
    
    word_to_replace = random.choice(words)
    
    synsets = wordnet.synsets(word_to_replace)
    if synsets:
        synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
        if synonyms:
            random_synonym = random.choice(synonyms)
            text_with_synonym = ' '.join([random_synonym if word == word_to_replace else word for word in words])
            t[attr] = text_with_synonym
    return t

def numerical_typos(t, attr):
    try:
        Err = typo.StrErrer(t[attr])
        method = random.choice(StrTypoNames)
        selected_method = getattr(Err, method)
        value = selected_method()   
        t[attr] = str(value)  
    except:
        pass
    return t

def text_typos(t, attr):
    try:
        Err = typo.IntErrer(t[attr])
        method = random.choice(IntTypoNames)
        selected_method = getattr(Err, method)
        value = selected_method()   
        t[attr] = int(value)
    except:
        pass
    return t
    
        
