#%% 
#############################
######## set-up part ########
#############################

import pandas as pd
import csv
import torch
from PIL import Image
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, AlignModel, AlignProcessor, BlipForImageTextRetrieval, BlipProcessor
import numpy as np
import faiss
import os
import urllib.request
from io import BytesIO
import json
from os import chdir as cd
import openai
from sklearn.metrics import precision_score, recall_score
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
from torch.nn.functional import normalize
import tiktoken

# define generative model, gpt-3.5-turbo
api_key = ""

openai.api_key = api_key
g_model_name = "gpt-3.5-turbo"


#%%
###############################
######## base function ########
###############################

# counting number of tokens
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# getting index
def get_index(model_ID, dataset_name):
    if model_ID == "openai/clip-vit-base-patch32":
        index = faiss.read_index(f"./faiss/clip-vit-base-patch32-{dataset_name}.index")
        data_index = pd.read_csv(f'./faiss/{dataset_name}_index.csv')
        return index, data_index
    
    elif model_ID == "kakaobrain/align-base":
        index = faiss.read_index(f"./faiss/align-base-{dataset_name}.index")
        data_index = pd.read_csv(f'./faiss/{dataset_name}_index.csv')
        return index, data_index

    elif model_ID == "Salesforce/blip-itm-base-coco":
        if dataset_name == 'coco':
            index = faiss.read_index(f"./faiss/blip-itm-base-coco-{dataset_name}.index")
            data_index = pd.read_csv(f'./faiss/{dataset_name}_index_blip.csv')
            return index, data_index
        
        else:
            index = faiss.read_index(f"./faiss/blip-itm-base-coco-{dataset_name}.index")
            data_index = pd.read_csv(f'./faiss/{dataset_name}_index.csv')
            return index, data_index              

# getting model info
def get_model_info(model_ID, device):
    if model_ID == "openai/clip-vit-base-patch32":
        model = CLIPModel.from_pretrained(model_ID).to(device)
        processor = CLIPProcessor.from_pretrained(model_ID)
        return model, processor
    
    elif model_ID == "kakaobrain/align-base":
        model = AlignModel.from_pretrained(model_ID).to(device)
        processor = AlignProcessor.from_pretrained(model_ID)
        return model, processor

    elif model_ID == "Salesforce/blip-itm-base-coco":
        model = BlipForImageTextRetrieval.from_pretrained(model_ID).to(device)
        processor = BlipProcessor.from_pretrained(model_ID)
        return model, processor

# getting raw dataset
def get_dataset(dataset_name):
    if dataset_name == 'coco':
        coco_full = []
        for i in range(1, 12):
            with open(f'./data/coco_train_captions/coco_train_{i}.json', 'r') as f:
                coco = json.load(f)
                coco_full.extend(coco['annotations'])
        coco_full = {'annotations': coco_full}
        dataset = coco_full.get('annotations', [])
        return dataset
    
    if dataset_name == 'flickr':
        dataset = pd.read_csv('./data/flickr_captions.txt', sep=',')
        return dataset
    
def get_dataset_2(dataset_name):
    if dataset_name == 'coco':
        dataset = pd.read_csv('./data/expansionnet.csv')
        return dataset
    
    if dataset_name == 'flickr':
        print("Warning! There is no Flickr dataset!")
        dataset = pd.read_csv('./data/flickr_captions.txt', sep=',')
        return dataset
    
# text embedding function
def get_single_text_embedding(generated_text):
    with torch.no_grad():
        if model_ID == "Salesforce/blip-itm-base-coco":
            text_encoder = model.text_encoder
            inputs = processor(text=generated_text, padding=True, truncation=True, return_tensors="pt")
            embedding = text_encoder(input_ids=inputs.input_ids, return_dict=True)[0]
            text_features = normalize(model.text_proj(embedding[:, 0, :]), dim=-1)
            embedding_as_np = text_features.cpu().numpy()
            
        else:
            inputs = processor(text=generated_text, padding=True, truncation=True, return_tensors="pt")
            embedding = model.get_text_features(**inputs)
            # convert the embeddings to numpy array
            embedding_as_np = embedding.cpu().numpy()
    return embedding_as_np

# getting image captions based on image indices
def get_captions(index_list, data_index):
    grouped_dict = {}
    index_list = [data_index.iloc[i].image_index for i in index_list]
    if dataset_name == 'coco':
        result = [(item['image_id'], item['caption']) for item in dataset if item['image_id'] in index_list]
    else:
        result = [(item['image_id'], item['caption']) for i, item in dataset.iterrows() if item['image_id'] in index_list]

    for key, value in result:
        if key in grouped_dict:
            grouped_dict[key].append(value)
        else:
            grouped_dict[key] = [value]
    result_list = [(key, values) for key, values in grouped_dict.items()]

    return result_list

def get_captions_2(index_list, data_index):
    grouped_dict = {}
    index_list = [data_index.iloc[i].image_index for i in index_list]
    
    result = [(item['image_id'], item['caption']) for i, item in dataset.iterrows() if item['image_id'] in index_list]

    for key, value in result:
        if key in grouped_dict:
            grouped_dict[key].append(value)
        else:
            grouped_dict[key] = [value]
    result_list = [(key, values) for key, values in grouped_dict.items()]

    return result_list

# show images
def get_images(dataset_name, image_ls):
    if dataset_name == 'coco':
        for image_path in image_ls:
            image_path = f'{image_path:012d}.jpg'
            image = img.imread(f'./data/coco_train_images/{image_path}')
            plt.imshow(image)
            plt.axis('off')
            plt.show()
    if dataset_name == 'flickr':
        for image_path in image_ls:
            image = img.imread(f'./data/flickr_images/{image_path}')
            plt.imshow(image)
            plt.axis('off')
            plt.show()
    if dataset_name == 'cc3m':
        for image_path in image_ls:
            image = img.imread(f'../etri_paper_v1/concpetual_imag/{image_path}')
            plt.imshow(image)
            plt.axis('off')
            plt.show()


# label 확인 함수
def check_label(row):
    if row['image_id'] == None:
        return 0
    elif row['input_text'] in str(row['gt']):
        return 1
    elif (row['input_text'].split(' ')[0] in str(row['gt'])) and (row['input_text'].split(' ')[1] in str(row['gt'])):
        return 1
    else: return 0

# 데이터프레임 단위 check_label() 적용 함수
def eval(result_df, gt_df):
    # gt_df['image_id'] = gt_df['image_id'].astype(str)
    # result_df['image_id'] = result_df['image_id'].astype(str) 
    full_df = pd.merge(result_df, gt_df, on='image_id', how='inner')
    full_df['answer'] = full_df.apply(check_label, axis=1)
    return full_df


#%%
##########################################
######## hypo experiment function ########
##########################################
    
# 1st turn situation generation part
def first_turn(model_ID, input_text, k):
    try: 
        prompt = f'''
        - Your role is to change the given word into a sentence in a specific situation consisting of specific nouns.
        This sentence will be used to find the most relevant image through the Vision-Language Model(VLM).
        You will change the word into a sentence through the process below.
            1. Check {input_text}.
            2. Check the dictionary definition of {input_text} in the Oxford dictionary.
            3. Remember at least three nouns that can express dictionary definitions.
            4. Select some of the nouns you remember and use them to describe a specific situation in one sentence.
            5. Modify the sentence in the form of an image description that preserves the meaning of {input_text} and is easy for the VLM to find.
            6. Output only the final sentence.
        Output just final sentence.

        - I'll show you this process through an example.
        - input_text : competitve
            1. input text : competitve
            2. The dictionary definition of 'competitive' is "Of, pertaining to, or characterized by competition; organized on the basis of competition."
            3. Some nouns that can illustrate this definition are ['sports', 'exam', 'player', 'baseball', 'study', 'soccer', 'boxing', 'game'].
            4. Among them, you can select ['sports', 'soccer', 'game'] to create the sentence "One sportman is playing in the soccer game."
            5. Revise the sentence to "One man wearing a uniform is kicking the ball with other players." to make it easier for VLM to find the image in the text and to preserve the meaning of 'competitive'.
            6. "One man wearing a uniform is kicking the ball with other players."
        - situation : "One man wearing a uniform is kicking the ball with other players."

        - Let's start.
        - input text : {input_text}
        - situation :
        '''
        
        response = openai.ChatCompletion.create(
            model=g_model_name,
            temperature = 0.5,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # extract situations
        hypo = response['choices'][0]['message']['content']
        embedded_text = get_single_text_embedding(hypo)
        
        if model_ID == 'kakaobrain/align-base':
            faiss.normalize_L2(embedded_text.reshape((1,640)))
        elif model_ID == 'Salesforce/blip-itm-base-coco':
            faiss.normalize_L2(embedded_text.reshape((1,256)))
        else:
            faiss.normalize_L2(embedded_text.reshape((1,512)))

        distances, indices = index.search(embedded_text, k)

        return indices[0].tolist(), hypo, distances[0, 0]
    
    except Exception as e:
        print('firstturn', e)
        return [999999999], 0, input_text

# multi-turn situation generation part
def multi_turn(input_text, result_list):
    try:
        # 현재 prompt는 k=1일 때를 기준으로 작성
        prompt = f'''
        - Your job is to check whether the caption is a semantically similar situation that can be replaced by the input text when expressed as a single word.
        The caption is the image_caption data of the (image, image_caption) pair.
        The sentence data that will be used instead of looking at the image to ensure that the most relevant image for the input text is retrieved.
        Answer can be either yes or no.

        - Here's an example
        - input text : competitive
        - caption : "A pitcher, batter and catcher in a baseball game."
        - answer : yes

        - input text : adventurous
        - caption : "A computer mouse is in front of two keyboards"
        - answer : no

        - Let's start
        - input text : {input_text}
        - caption : {result_list[0][1]}
        - answer : 
        '''
        
        response = openai.ChatCompletion.create(
            model=g_model_name,
            temperature = 0.5,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=4
        )
    
        # extract situations
        generated_text = response['choices'][0]['message']['content']
        # logprobs = response['choices'][0]['logprobs']['content']
        logprob = response['choices'][0]['logprobs']['content'][0]['logprob']
        prob = np.round(np.exp(logprob)*100, 2)

        if ('yes' in generated_text or 'Yes' in generated_text) and (prob >= 70):
            return input_text, result_list[0][0]
        else: 
            return "retry"
            
    except Exception as e:
        print('multiturn', e)
        return input_text, [999999999]

# re-generation function
def re_first_turn(model_ID, input_text, hypo, num):
    try: 
        num += 1
        prompt = f'''
                - Your role is to change the given word into a sentence in a specific situation consisting of specific nouns.
                This sentence will be used to find the most relevant image through the Vision-Language Model(VLM).
                You will change the word into a sentence through the process below. Output just final sentence.
                    1. Check {input_text}.
                    2. Check the dictionary definition of {input_text} in the Oxford dictionary.
                    3. Remember at least three nouns that can express dictionary definitions.
                    4. Select some of the nouns you remember and use them to describe a specific situation in one sentence.
                    5. If the context of the sentence is {hypo}, change it to another context.
                    6. Modify the sentence in the form of an image description that preserves the meaning of {input_text} and is easy for the VLM to find.
                    7. Output only the final sentence.

                - I'll show you this process through an example.
                - input_text : competitve
                - exclude : ["One soccer player is kicking the ball in the field.", "A person is running while listening to music."]
                    1. input text : competitve
                    2. The dictionary definition of 'competitive' is "Of, pertaining to, or characterized by competition; organized on the basis of competition."
                    3. Some nouns that can illustrate this definition are ['sports', 'exam', 'player', 'baseball', 'study', 'soccer', 'boxing', 'game'].
                    4. Among them, you can select ['sports', 'soccer', 'game'] to create the sentence "One sportman is playing in the soccer game."
                    5. Since "One sportman is playing in the soccer game." is similar to the situation with "One soccer player is kicking the ball in the field." in the exclude list, 
                        Select the nouns ['exam', 'study'] and create the sentence "Many students are taking exams."
                    6. Revise the sentence to "Many students are writing the answer on the paper in the classroom." to make it easier for VLM to find the image in the text and to preserve the meaning of 'competitive'.
                    7. "Many students are writing the answer on the paper in the classroom."
                - situation : "Many students are writing the answer on the paper in the classroom."

                - Let's start.
                - input text : {input_text}
                - exclude : [{hypo}]
                - situation :
                '''

        response = openai.ChatCompletion.create(
            model=g_model_name,
            temperature = 0.5,
            messages=[{"role": "user", "content": prompt}]
        )
    
        # extract situations
        re_hypo = response['choices'][0]['message']['content']
        embedded_text = get_single_text_embedding(re_hypo)

        if model_ID == 'kakaobrain/align-base':
            faiss.normalize_L2(embedded_text.reshape((1,640)))
        elif model_ID == 'Salesforce/blip-itm-base-coco':
            faiss.normalize_L2(embedded_text.reshape((1,256)))
        else:
            faiss.normalize_L2(embedded_text.reshape((1,512)))

        distances, indices = index.search(embedded_text, k)
        
        return indices[0].tolist(), re_hypo, distances[0, 0], num 
    
    except Exception as e:
        print(e)
        return [999999999], input_text, num

# hypo multi-turn experiment
def experiment(model_ID, input_ls, k, max_iter):

    result = []

    for input_text in input_ls:
        first_turn_start = time.time()
        index_list, hypo, distances = first_turn(model_ID, input_text, k)
        result_list = get_captions_2(index_list, data_index)
        first_turn_end = time.time()

        multi_turn_start = time.time()
        num = 0
        # while multi_turn(input_text, result_list) == "retry":
        #     index_list, re_hypo, distances, num = re_first_turn(model_ID, input_text, hypo, num)
        #     hypo = hypo + ',' + re_hypo
        #     result_list = get_captions(index_list, data_index)
        #     if num > max_iter+1 :
        #         break
        while num < max_iter :
            if multi_turn(input_text, result_list) != "retry":
                break
            index_list, re_hypo, distances, num = re_first_turn(model_ID, input_text, hypo, num)
            hypo = hypo + ',' + re_hypo
            result_list = get_captions_2(index_list, data_index)
        multi_turn_end = time.time()

        first_turn_time = first_turn_end - first_turn_start
        multi_turn_time = multi_turn_end - multi_turn_start
        result_list = [(input_text, hypo, distances, item[0], item[1], num, first_turn_time, multi_turn_time) for item in result_list]
        result.append(result_list)

    result_df = pd.DataFrame()
    
    
    # 데이터프레임에 데이터 추가
    for sublist in result:
        for item in sublist:
            sub_df = pd.DataFrame({
                'input_text' : item[0], 
                'hypo' : item[1], 
                'distances' : item[2],
                'image_id': [item[3]], 
                'captions': [item[4]], 
                'iter_num' : item[5],
                'first_turn_time' : item[6],
                'multi_turn_time' : item[7],
                'max_iter' : max_iter,
                'output_token' : len(encoding.encode(item[1]))
                })
            result_df = pd.concat([result_df, sub_df], ignore_index=True)
            result_df.columns=['input_text', 'hypo', 'distances', 'image_id', 'captions', 'iter_num', 'first_turn_time', 'multi_turn_time', 'max_iter', 'output_token']
    return result_df


#%%
        
############################
######## experiment ########
############################

### 실험 변수 설정 ###
        
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Define the model ID, dataset
CLIP = "openai/clip-vit-base-patch32"
ALIGN = "kakaobrain/align-base"
BLIP = "Salesforce/blip-itm-base-coco"

model_ID = BLIP
model_id = 'blip' # csv 저장용
dataset_name = 'coco'

# Get model, processor & tokenizer
model, processor = get_model_info(model_ID, device)
index, data_index = get_index(model_ID, dataset_name)
dataset = get_dataset_2(dataset_name)

# ground-truth setting
gt_df = pd.read_csv('./data/gt/coco.csv')
# gt_df.columns = ['image_id', 'caption', 'gt_others', 'gt_noun']

# pos = 'noun'

# top-k
k = 1

# maximum number of iteration
max_iter = 5


# %%

def origin_exp(model_ID, input_ls, k):

    try:
        result = []

        for input_text in input_ls:

            start_time = time.time()

            embedded_text = get_single_text_embedding(input_text)
            if model_ID == 'kakaobrain/align-base':
                faiss.normalize_L2(embedded_text.reshape((1,640)))
            elif model_ID == 'Salesforce/blip-itm-base-coco':
                faiss.normalize_L2(embedded_text.reshape((1,256)))
            else:
                faiss.normalize_L2(embedded_text.reshape((1,512)))

            distances, indices = index.search(embedded_text, k)
            index_list = indices[0].tolist()
            result_list = get_captions(index_list, data_index)

            end_time = time.time()
            origin_time = end_time - start_time

            result_list = [(input_text, item[0], item[1], distances[0, 0], origin_time) for item in result_list]
            result.append(result_list)
        
            
        result_df = pd.DataFrame()


        # 데이터프레임에 데이터 추가
        for sublist in result:
            for item in sublist:
                sub_df = pd.DataFrame({
                    'input_text' : item[0], 
                    'image_id': [item[1]], 
                    'captions': [item[2]], 
                    'distances' : item[3],
                    'origin_time' : item[4],
                    })
                result_df = pd.concat([result_df, sub_df], ignore_index=True)
                result_df.columns=['input_text', 'image_id', 'captions', 'distances', 'origin_time']

        return result_df
    
    except Exception as e:
        print(e) 




#%%

def multi_turn_ver(input_text, result_list):
    try:
        # 현재 prompt는 k=1일 때를 기준으로 작성
        prompt = f'''
        - Your job is to check whether the caption is a semantically similar situation that can be replaced by the input text when expressed as a single word.
        The caption is the image_caption data of the (image, image_caption) pair.
        The sentence data that will be used instead of looking at the image to ensure that the most relevant image for the input text is retrieved.
        Answer can be either yes or no.

        - Here's an example
        - input text : competitive
        - caption : "A pitcher, batter and catcher in a baseball game."
        - answer : yes

        - input text : adventurous
        - caption : "A computer mouse is in front of two keyboards"
        - answer : no

        - Let's start
        - input text : {input_text}
        - caption : {result_list[0][1]}
        - answer : 
        '''
        
        response = openai.ChatCompletion.create(
            model=g_model_name,
            temperature = 0.5,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=4
        )
    
        # extract situations
        generated_text = response['choices'][0]['message']['content']
        # logprobs = response['choices'][0]['logprobs']['content']
        logprob = response['choices'][0]['logprobs']['content'][0]['logprob']
        prob = np.round(np.exp(logprob)*100, 2)

        if ('yes' in generated_text or 'Yes' in generated_text) and (prob >= 70):
            return input_text, result_list[0][0]
        else: 
            return "retry"
            
    except Exception as e:
        print('multiturn', e)
        return input_text, [999999999]

def origin_exp_ver(model_ID, input_ls, k):

    try:
        result = []

        for input_text in input_ls:

            start_time = time.time()

            embedded_text = get_single_text_embedding(input_text)
            if model_ID == 'kakaobrain/align-base':
                faiss.normalize_L2(embedded_text.reshape((1,640)))
            elif model_ID == 'Salesforce/blip-itm-base-coco':
                faiss.normalize_L2(embedded_text.reshape((1,256)))
            else:
                faiss.normalize_L2(embedded_text.reshape((1,512)))

            distances, indices = index.search(embedded_text, k)

            x = 0
            index_list = indices[0][x:x+1].tolist()
            result_list = get_captions(index_list, data_index)

            while multi_turn_ver(input_text, result_list) == "retry":
                x += 1
                index_list = indices[0][x:x+1].tolist()
                result_list = get_captions(index_list, data_index)
                # if x > 3:
                #     break

            end_time = time.time()
            origin_time = end_time - start_time

            result_list = [(input_text, item[0], item[1], distances[0, 0], origin_time) for item in result_list]
            result.append(result_list)
        
        result_df = pd.DataFrame()

        # 데이터프레임에 데이터 추가
        for sublist in result:
            for item in sublist:
                sub_df = pd.DataFrame({
                    'input_text' : item[0], 
                    'image_id': [item[1]], 
                    'captions': [item[2]], 
                    'distances' : item[3],
                    'origin_time' : item[4],
                    })
                result_df = pd.concat([result_df, sub_df], ignore_index=True)
                result_df.columns=['input_text', 'image_id', 'captions', 'distances', 'origin_time']

        return result_df
    
    except Exception as e:
        print(e) 
#%%

pos = 'noun'
if pos == 'adj':   
    input_ls = ['nervous', 'painful', 'paranoid', 'faithful', 'furious', 'refreshing', 'exhausting', 'cooperative', 'holy', 'fantastic']
elif pos == 'adv':
    input_ls = ['competitively', 'technologically', 'joyously', 'fearfully', 'extravagantly', 'offensively', 'silently', 'historically', 'speedily', 'haphazardly']
elif pos == 'verb':
    # input_ls = ['produce', 'shower', 'knock', 'record', 'interview', 'juggle', 'complete', 'clean', 'pretend', 'eat']
    input_ls = ['cut', 'drink', 'stand', 'knock', 'hit', 'drive', 'paddle', 'perform', 'decorate', 'jump']
elif pos == 'noun':
    input_ls = ['girl', 'giraffe', 'oven', 'bed', 'dogs', 'room', 'orange', 'pizza', 'bike', 'monitor']
elif pos == 'p':
    input_ls = ['cloudy sky', 'refreshing carrot', 'snowy mountain', 'cooperative player', 'fantastic view', 'joyously walk', 'competitively game', 'speedily delivery', 'historically build', 'faithful dog' ]


print(f"모델: {model_id}, 데이터셋: {dataset_name}, 품사: {pos}")
for i in range(5):
    result_df = experiment(model_ID, input_ls, k, max_iter)
    after_df = eval(result_df,gt_df)
    # 평균 확인
    print(after_df['answer'].mean())

pos = 'verb'
if pos == 'adj':   
    input_ls = ['nervous', 'painful', 'paranoid', 'faithful', 'furious', 'refreshing', 'exhausting', 'cooperative', 'holy', 'fantastic']
elif pos == 'adv':
    input_ls = ['competitively', 'technologically', 'joyously', 'fearfully', 'extravagantly', 'offensively', 'silently', 'historically', 'speedily', 'haphazardly']
elif pos == 'verb':
    # input_ls = ['produce', 'shower', 'knock', 'record', 'interview', 'juggle', 'complete', 'clean', 'pretend', 'eat']
    input_ls = ['cut', 'drink', 'stand', 'knock', 'hit', 'drive', 'paddle', 'perform', 'decorate', 'jump']
elif pos == 'noun':
    input_ls = ['girl', 'giraffe', 'oven', 'bed', 'dogs', 'room', 'orange', 'pizza', 'bike', 'monitor']
elif pos == 'p':
    input_ls = ['cloudy sky', 'refreshing carrot', 'snowy mountain', 'cooperative player', 'fantastic view', 'joyously walk', 'competitively game', 'speedily delivery', 'historically build', 'faithful dog' ]

print(f"모델: {model_id}, 데이터셋: {dataset_name}, 품사: {pos}")
for i in range(5):
    result_df = experiment(model_ID, input_ls, k, max_iter)
    after_df = eval(result_df,gt_df)
    # 평균 확인
    print(after_df['answer'].mean())

pos = 'adj'
if pos == 'adj':   
    input_ls = ['nervous', 'painful', 'paranoid', 'faithful', 'furious', 'refreshing', 'exhausting', 'cooperative', 'holy', 'fantastic']
elif pos == 'adv':
    input_ls = ['competitively', 'technologically', 'joyously', 'fearfully', 'extravagantly', 'offensively', 'silently', 'historically', 'speedily', 'haphazardly']
elif pos == 'verb':
    # input_ls = ['produce', 'shower', 'knock', 'record', 'interview', 'juggle', 'complete', 'clean', 'pretend', 'eat']
    input_ls = ['cut', 'drink', 'stand', 'knock', 'hit', 'drive', 'paddle', 'perform', 'decorate', 'jump']
elif pos == 'noun':
    input_ls = ['girl', 'giraffe', 'oven', 'bed', 'dogs', 'room', 'orange', 'pizza', 'bike', 'monitor']
elif pos == 'p':
    input_ls = ['cloudy sky', 'refreshing carrot', 'snowy mountain', 'cooperative player', 'fantastic view', 'joyously walk', 'competitively game', 'speedily delivery', 'historically build', 'faithful dog' ]

print(f"모델: {model_id}, 데이터셋: {dataset_name}, 품사: {pos}")
for i in range(5):
    result_df = experiment(model_ID, input_ls, k, max_iter)
    after_df = eval(result_df,gt_df)
    # 평균 확인
    print(after_df['answer'].mean())

pos = 'adv'
if pos == 'adj':   
    input_ls = ['nervous', 'painful', 'paranoid', 'faithful', 'furious', 'refreshing', 'exhausting', 'cooperative', 'holy', 'fantastic']
elif pos == 'adv':
    input_ls = ['competitively', 'technologically', 'joyously', 'fearfully', 'extravagantly', 'offensively', 'silently', 'historically', 'speedily', 'haphazardly']
elif pos == 'verb':
    # input_ls = ['produce', 'shower', 'knock', 'record', 'interview', 'juggle', 'complete', 'clean', 'pretend', 'eat']
    input_ls = ['cut', 'drink', 'stand', 'knock', 'hit', 'drive', 'paddle', 'perform', 'decorate', 'jump']
elif pos == 'noun':
    input_ls = ['girl', 'giraffe', 'oven', 'bed', 'dogs', 'room', 'orange', 'pizza', 'bike', 'monitor']
elif pos == 'p':
    input_ls = ['cloudy sky', 'refreshing carrot', 'snowy mountain', 'cooperative player', 'fantastic view', 'joyously walk', 'competitively game', 'speedily delivery', 'historically build', 'faithful dog' ]

print(f"모델: {model_id}, 데이터셋: {dataset_name}, 품사: {pos}")
for i in range(5):
    result_df = experiment(model_ID, input_ls, k, max_iter)
    after_df = eval(result_df,gt_df)
    # 평균 확인
    print(after_df['answer'].mean())
# %%
