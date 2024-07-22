# gpt-3.5를 사용하는 dataset 생성 code

# -*- coding: utf-8 -*-

#%%
import pandas as pd
import openai
import time
import json
import os


#%%
# model 설정
api_key = ""
openai.api_key = api_key
model_name = "gpt-3.5-turbo"


#%%
# coco caption 불러오기

with open('./coco_train_captions/coco_train_11.json', 'r') as f:
     coco = json.load(f)


#%%
result_list = []

for i in range(len(coco['annotations'])):
    try:
        caption = coco['annotations'][i]['caption']
        time.sleep(10)

        user_msg = f"""
        Your job is to supplement the dataset, which should be used as groundtruth with minimal bias. The caption you'll see describes the image.
        View the caption and label them with additional words.
        You can view the caption and select one label from the label set that best describe the image. The label set looks like this:
        abstract label set : [nervous, painful, paranoid, faithful, furious, refreshing, exhausting, cooperative, holy, fantastic, competitively, technologically, joyously, fearfully, extravagantly, offensively, silently, historically, speedily, haphazardly, introduce, spoil, salvage, applaud, pray, stink, fulfill, overcome, negotiate, shuffle]
        
        If there is no related label with caption, just output 'none' for the answer.
        Now let me show you an example.
        
        - caption: The clock for sale has birds displayed on it.
        - label: silently
        
        - caption: {caption}
        - label: """
        
        response = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.5,
                messages=[{"role": "user", "content": user_msg}
                ])
        
        result = (coco['annotations'][i]['image_id'], response['choices'][0]['message']['content'])
        result_list.append(result)

        print(i, result)

    except:
        continue
        


# %%
