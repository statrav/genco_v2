'''
Motivation 파트
image t-sne
'''


#%%

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
import json
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import pandas as pd
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import functional as F
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from datasets import load_dataset
from collections import OrderedDict
import faiss
import os
import urllib.request
from io import BytesIO
from matplotlib import image as mpimg


#%%
coco_faiss = faiss.read_index("faiss/clip-vit-base-patch32-coco.index")
coco_index = pd.read_csv('faiss/coco_index_original.csv')


# %%
model_ID = 'openai/clip-vit-base-patch32'

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_ID).to(device)
processor = CLIPProcessor.from_pretrained(model_ID)
tokenizer = CLIPTokenizer.from_pretrained(model_ID)


#%%
"""
이미지 검색 함수 정의
한 query 당 이미지 k개 검색
input: query list
return: query에 대한 검색 결과 이미지의 경로
"""

def retrieve_images(query_list):
    image_paths = []
    for query in query_list:
        inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
        embedding = model.get_text_features(**inputs)

        query_vector = embedding.cpu().detach().numpy()

        faiss.normalize_L2(query_vector.reshape((1,512)))  ## 검색 전 무조건 normalize!
        # print(query_vector)
        distances, indices = coco_faiss.search(query_vector, k)

        indices_list = list(indices[0])
        path_list = [coco_index['image_index'].loc[path] for path in indices_list]
        image_paths += [f'coco_train_images/{name}' for name in path_list]

    return image_paths


#%%
"""
이미지 임베딩 함수 정의
query list에 대한 검색 결과 이미지를 임베딩
return: 검색 결과 이미지들의 embedded vectors
"""

def embed_images(query_list):
    image_paths = retrieve_images(query_list)
    
    embeddings = []
    for image_path in image_paths:
        image = read_image(image_path)
        image = F.to_pil_image(image)
        inputs = processor(text=None, images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = model.get_image_features(**inputs).cpu().numpy()
        embeddings.append(image_embedding)
    return np.concatenate(embeddings, axis=0)


#%%
"""
tsne plotting 함수 정의
input: query list
output: tsne plot
"""

def tsne(query_list):
    image_embeddings = embed_images(query_list)
    image_labels = [[query]*k for query in query_list]

    tsne = TSNE(n_components=2, random_state=0)
    image_embeddings_2d = tsne.fit_transform(image_embeddings)

    # colors = plt.cm.Dark2(np.linspace(0, 1, len(query_list)))

    # 시각화
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(query_list):
        indices = list(range(i*30, i*30+30))
        x, y = image_embeddings_2d[indices].T
        plt.scatter(x, y, label=label)

    # for i, label in enumerate(unique_labels):
    #     plt.annotate(label, (0, 0), color=colors[i])  # 레전드 표시 위치

    plt.legend()
    plt.show()


#%%
"""
intra cluster distance 계산 함수 정의
input: image embedding vector, label (정수)
output: intra cluster distance의 평균
"""

def intra_cluster_distance(data, labels):
    unique_labels = np.unique(labels)
    total_distance = 0

    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            pairwise_distances = cdist(cluster_points, cluster_points, metric='euclidean')
            avg_distance = np.mean(pairwise_distances[np.triu_indices(len(cluster_points), k=1)])
            total_distance += avg_distance

    return total_distance / len(unique_labels)


#%%
"""
inter cluster distance 계산 함수 정의
input: image embedding vector, label (정수)
output: inter cluster distance의 평균
"""

def inter_cluster_distance(data, labels):
    unique_labels = np.unique(labels)  # 군집 레이블의 고유값 추출
    centroids = []

    # 군집별 중심점 계산
    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # 중심점 간의 거리 측정
    pairwise_distances = cdist(centroids, centroids, metric='euclidean')
    
    # 군집 간 거리의 평균 반환
    return np.mean(pairwise_distances)


#%%
query_list = ['running cat', 'sleeping cat', 'running cow'] # tsne에 사용할 query
# query_list = ['bike', 'apple', 'bird', 'steak', 'telephone', 'cow', 'rail', 'subway', 'cat']
k = 10 # 검색 top-k


#%%
tsne(query_list)


#%%
image_embeddings = embed_images(query_list)
image_labels = [query for query in list(range(9)) for _ in range(30)]


#%%
print(intra_cluster_distance(image_embeddings, image_labels))
print(inter_cluster_distance(image_embeddings, image_labels))

# %%
