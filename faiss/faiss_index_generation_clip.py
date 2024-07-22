"""
faiss index 생성 코드
flickr8k 기준
"""

#%%
import pandas as pd
from PIL import Image
import torch
from datasets import load_dataset
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
import faiss
import os
import urllib.request
from io import BytesIO
import json


#%%
# CLIP 모델 불러오기
def get_model_info(model_ID, device):
# Save the model to device
	model = CLIPModel.from_pretrained(model_ID).to(device)
 	# Get the processor
	processor = CLIPProcessor.from_pretrained(model_ID)
# Get the tokenizer
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
       # Return model, processor & tokenizer
	return model, processor, tokenizer

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
model, processor, tokenizer = get_model_info(model_ID, device)


#%%
# image embedding 함수
def get_single_image_embedding(my_image):
    try:
        image = processor(text=None, images=my_image, return_tensors="pt")["pixel_values"].to(device)
        embedding = model.get_image_features(image)

        # convert the embeddings to numpy array
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_all_images_embedding(df, img_column):
    df["img_embeddings"] = df[str(img_column)].apply(get_single_image_embedding)
    return df


#%%
# faiss hyperparameter 지정
dimension = 512
index = faiss.IndexFlatIP(dimension)
index = faiss.IndexIDMap2(index)


#%%
# 추후 faiss index와 image를 매칭하기 위한 csv 파일 생성
file_list = os.listdir('./flickr_images')
flickr = pd.DataFrame(file_list, columns=['image_path'])
flickr.to_csv('flickr_index.csv')


#%%
def get_image(image_path):
    try:
        image = Image.open(f'./flickr_images/' + image_path)
    except:
        image = None
    return image

#%%
flickr_raw = flickr.copy()

for i in range(0, len(flickr_raw)//400):
    flickr = flickr_raw.iloc[400*i:400*(i+1)]

    flickr['image'] = flickr['image_path'].apply(get_image)
    df_clean = flickr.dropna()
    image_data_df = get_all_images_embedding(df_clean, "image")
    image_data_df = image_data_df.dropna()

    for idx, row in image_data_df.iterrows():
        faiss.normalize_L2(row.img_embeddings)
        index.add_with_ids(row.img_embeddings, idx)
    print(i,'th iter 완료')

final_chunk = flickr.iloc[len(flickr_raw)//400:]
final_chunk['image'] = final_chunk['image_path'].apply(get_image)
df_clean = final_chunk.dropna()
image_data_df = get_all_images_embedding(df_clean, "image")

for idx, row in image_data_df.iterrows():
    faiss.normalize_L2(row.img_embeddings)
    index.add_with_ids(row.img_embeddings, idx)
print('마지막 chunk까지 완료!')

faiss.write_index(index,"clip-vit-base-patch32-flickr.index")
#%%
# 인덱스 개수 확인
index.ntotal

