#%%
#####################################
############ ALIGN_FAISS ############
#####################################

from PIL import Image
from transformers import AutoProcessor, AlignModel
import torch
import os
import pandas as pd
import numpy as np
import faiss
import json


#%%
# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Blip Model
model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)
processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

#%%
# faiss hyperparameter 지정
dimension = 640
index = faiss.IndexFlatIP(dimension)
index = faiss.IndexIDMap2(index)

cc3m = pd.read_csv('./faiss/cc3m_index.csv')

#%%
# image embedding 함수
def get_single_image_embedding(my_image):
    try:
        inputs = processor(images=my_image, return_tensors="pt")
        inputs.to(device)
        embedding = model.get_image_features(**inputs)
        embedding.to(device)

        # convert the embeddings to numpy array
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_all_images_embedding(df, img_column):
    df["img_embeddings"] = df[str(img_column)].apply(get_single_image_embedding)
    return df

def get_image(image_path):
    try:
        image = Image.open(f'../etri_paper_v1/conceptual_img/' + image_path)
    except:
        image = None
    return image

#%%
cc3m_raw = cc3m.copy()

for i in range(0, len(cc3m_raw)//400):
    cc3m = cc3m_raw.iloc[400*i:400*(i+1)]

    cc3m['image'] = cc3m['image_path'].apply(get_image)
    df_clean = cc3m.dropna()
    image_data_df = get_all_images_embedding(df_clean, "image")
    image_data_df = image_data_df.dropna()

    for idx, row in image_data_df.iterrows():
        faiss.normalize_L2(row.img_embeddings)
        index.add_with_ids(row.img_embeddings, idx)
    print(i,'th iter 완료')

final_chunk = cc3m.iloc[len(cc3m_raw)//400:]
final_chunk['image'] = final_chunk['image_path'].apply(get_image)
df_clean = final_chunk.dropna()
image_data_df = get_all_images_embedding(df_clean, "image")
#%%
image_data_df = image_data_df.dropna()
for idx, row in image_data_df.iterrows():
    faiss.normalize_L2(row.img_embeddings)
    index.add_with_ids(row.img_embeddings, idx)
print('마지막 chunk까지 완료!')

faiss.write_index(index,"./align-bas-cc3m.index")

#dimension check 방법 : len(img_embeddings[0])
#%%
index.ntotal

#%%

## index csv 만드는 코드
file_list = os.listdir('../etri_paper_v1/conceptual_img')
flickr = pd.DataFrame(file_list, columns=['image_path'])
flickr.to_csv('./faiss/cc3m_index.csv')
# %%
  