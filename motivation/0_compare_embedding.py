'''
Motivation part
품사 기준으로 matrix 거리 변화 측정
'''

#%%
import pandas as pd
import csv
import torch
from PIL import Image
from datasets import load_dataset
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from numpy.linalg import norm
import faiss
import os
import urllib.request
from io import BytesIO
import json
from os import chdir as cd
import random


# %%
'''
VL model setting
'''
def get_model_info(model_ID, device):
    if model_ID == "openai/clip-vit-base-patch32":
        model = CLIPModel.from_pretrained(model_ID).to(device)
        processor = CLIPProcessor.from_pretrained(model_ID)
        return model, processor
    
device = "cpu"

# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
model, processor = get_model_info(model_ID, device)


#%%
'''
구/절 형태 실험을 위한 random 생성
'''
# noun list
nouns = ['kids', 'table', 'window', 'computer', 'flowers',
         'toothbrushes', 'television', 'bedroom', 'person', 'tennis']

# verb list
verbs = ['play', 'lay', 'sit', 'eat', 'walk',
         'smile', 'hit', 'use', 'hold', 'wear']

# adjective list
adjs = ['']

# 문장 생성 함수
def generate_term():
    noun_list = random.sample(nouns, 2)
    verb_list = random.sample(verbs, 2)
    return [f"{noun_list[0]} {verb_list[0]}", f"{noun_list[1]} {verb_list[0]}", f"{noun_list[0]} {verb_list[1]}"]

# 30개의 adj+noun 생성
noun_verb_list = []
for _ in range(30):
    noun_verb_list.append(generate_term())


#%%
'''
문장 생성 -> chatgpt 써서 생성

Make 30 sentences with the given words.
You do not have to use all words at once.

nouns = ['kids', 'table', 'window', 'computer', 'flowers',
         'toothbrushes', 'television', 'bedroom', 'person', 'tennis']

verbs = ['play', 'lay', 'sit', 'eat', 'walk',
         'smile', 'hit', 'use', 'hold', 'wear']
'''

noun_verb_sentence_list = [["The kids play tennis in the backyard.",  "The kids play tennis in the beach.", "The kids rest in the backyard."], 
                           ["Flowers bloom outside the bedroom window.",  "Kids bloom outside the bedroom window.", "Flowers sway outside the bedroom window."],
                           ["The person smiles while using the computer.", "The person smiles while using the piano.", "The person lays down while using the computer."],
                           ["Kids sit around the table for dinner.", "Pilots sit around the table for dinner.", "Kids walk around the table for dinner."], 
                           ["A television sits atop the table in the living room.", "One guy sits atop the table in the living room.", "A television falls down the table in the living room."],
                           ["The person holds a bouquet of flowers.", "Dog holds a bouquet of flowers.", "The person throws a bouquet of flowers."],
                           ["Kids play with their toys on the bedroom floor.", "Cats play with their toys on the bedroom floor.", "Kids organize their toys on the bedroom floor."],
                           ["The person lays down on the bed after a long day.", "The person lays down on the couch after a long day.", "The person jumps on the couch after a long day."],
                           ["Toothbrushes sit neatly on the bathroom table.", "The boy sits neatly on the bathroom table.", "Toothbrushes hang neatly on the bathroom table."],
                           ["Kids walk to school together in the morning.", "Kids walk to school together in the night.", "Kids run to school together in the morning."]]

#%%
'''
VL model에서 cosine similarity 및 l2 distance 계산 함수
input: q1 ~ q3
return: q1q2_cos, q1q3_cos, q1q2_l2, q1q3_l2
'''

def calculate_vl(q_list):
    with torch.no_grad():
        inputs = processor(text=q_list, padding=True, truncation=True, return_tensors="pt")
        vectors = model.get_text_features(**inputs)
        np_vectors = vectors.cpu().numpy()
        q1_vector = np_vectors[0]
        q2_vector = np_vectors[1]
        q3_vector = np_vectors[2]
    return q_list + [np.dot(q1_vector, q2_vector.T) / (norm(q1_vector) * norm(q2_vector.T)), np.dot(q1_vector, q2_vector.T) / (norm(q1_vector) * norm(q3_vector.T)), norm(q1_vector - q2_vector), norm(q1_vector - q3_vector)]


#%%
fields = ['q1', 'q2', 'q3', 'q1q2_cos', 'q1q3_cos', 'q1q2_l2', 'q1q3_l2']
result = [calculate_vl(q) for q in noun_verb_sentence_list]

with open('../motivation/compare_embedding_noun_verb_sent.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(result)

# %%
# 통계량 확인해볼까~
    
df = pd.read_csv('../motivation/compare_embedding_noun_verb_sent.csv')

print(df['q1q2_cos'].mean(), df['q1q3_cos'].mean())
print(df['q1q2_cos'].std(), df['q1q3_cos'].std())
print(df['q1q2_l2'].mean(), df['q1q3_l2'].mean())
print(df['q1q2_l2'].std(), df['q1q3_l2'].std())

# %%
