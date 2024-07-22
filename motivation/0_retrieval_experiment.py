# motivation experiment (COCO- CLIP)
# 기존 모델의 한계점을 찾기 위해, initial query 그대로 검색 진행

#%%
import pandas as pd
from PIL import Image
import torch
from datasets import load_dataset
from collections import OrderedDict
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import numpy as np
import faiss
import os
import urllib.request
from io import BytesIO
import json
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


#%%
index = faiss.read_index("../faiss/clip-vit-base-patch32-coco.index")
index.ntotal

#%%
coco_index = pd.read_csv('../faiss/coco_index_original.csv')
# %%
def experiment_clip(query, k, model_ID):
    device = "cpu"
    model = CLIPModel.from_pretrained(model_ID).to(device)
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)

    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
    embedding = model.get_text_features(**inputs)

    query_vector = embedding.cpu().detach().numpy()

    faiss.normalize_L2(query_vector.reshape((1,512)))  ## 검색 전 무조건 normalize!
    # print(query_vector)
    distances, indices = index.search(query_vector, k)

    for i in indices[0]:
         image = coco_index.iloc[i].image_index
         image_path = f'../coco_train_images/{image}'
         image = mpimg.imread(image_path)
         plt.imshow(image)
         plt.axis('off')
         plt.show()

    print(distances, indices)

# %%
def experiment_align(query, k, model_ID):
    device = "cpu"
    model = AlignModel.from_pretrained(model_ID).to(device)
    processor = AlignProcessor.from_pretrained(model_ID)

    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
    embedding = model.get_text_features(**inputs)

    query_vector = embedding.cpu().detach().numpy()

    faiss.normalize_L2(query_vector.reshape((1,640)))  ## 검색 전 무조건 normalize!
    # print(query_vector)
    distances, indices = index.search(query_vector, k)

    # for i in indices[0]:
    #      image_path = './coco_train_images/image_' + str(i) +'.jpg'
    #      image = mpimg.imread(image_path)
    #      plt.imshow(image)
    #      plt.axis('off')
    #      plt.show()

    print(distances, indices)
# %%
experiment_clip("The kids rest in the backyard.", 5, 'openai/clip-vit-base-patch32')
# %%
