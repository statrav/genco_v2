"""
caption distribution (pos 기준)
- vl dataset
    1. COCO captions
    2. Flickr8k
- language only dataset
    1. BookCorpus
    2.
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import json
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from datasets import load_dataset
import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# %%
def pos_plot(full_caption):
    words = word_tokenize(full_caption)
    tagged_words = nltk.pos_tag(words)
    tags = [tag for word, tag in tagged_words]
    tag_freq = Counter(tags)
    
    # NN, JJ, VB의 활용형을 NN, JJ, VB로 mapping
    tag_freq['JJ'] = tag_freq['JJ'] + tag_freq['JJR'] + tag_freq['JJS']
    tag_freq['VB'] = tag_freq['VB'] + tag_freq['VBD'] + tag_freq['VBG'] + tag_freq['VBN'] + tag_freq['VBP'] + tag_freq['VBZ']
    tag_freq['NN'] = tag_freq['NN'] + tag_freq['NNS'] + tag_freq['NNP'] + tag_freq['NNPS']
    
    # 문장부호 삭제
    del tag_freq['.']
    del tag_freq[',']

    # 기타 품사 count
    sub_tag_freq = {key: value for key, value in tag_freq.items() if key not in {'VB', 'NN', 'JJ', 'RB', 'IN', 'DT'}}
    # print(sub_tag_freq)

    # 최종적인 품사 count dictionary 생성
    final_tag_freq = {key: value for key, value in tag_freq.items() if key in {'VB', 'NN', 'JJ', 'RB', 'IN', 'DT'}}
    final_tag_freq['etc'] = sum(sub_tag_freq.values())

    x = list(final_tag_freq.keys())
    y = list(final_tag_freq.values())

    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.title('figure')
    plt.xlabel('pos')
    plt.ylabel('freq')
    plt.show()

    print(final_tag_freq)


# %%
def pos_tag(full_caption):
    words = word_tokenize(full_caption)
    tagged_words = nltk.pos_tag(words)
    tags = [tag for word, tag in tagged_words]
    tag_freq = Counter(tags)
    
    # NN, JJ, VB의 활용형을 NN, JJ, VB로 mapping
    tag_freq['JJ'] = tag_freq['JJ'] + tag_freq['JJR'] + tag_freq['JJS']
    tag_freq['VB'] = tag_freq['VB'] + tag_freq['VBD'] + tag_freq['VBG'] + tag_freq['VBN'] + tag_freq['VBP'] + tag_freq['VBZ']
    tag_freq['NN'] = tag_freq['NN'] + tag_freq['NNS'] + tag_freq['NNP'] + tag_freq['NNPS']
    
    # 문장부호 삭제
    del tag_freq['.']
    del tag_freq[',']

    # 기타 품사 count
    sub_tag_freq = {key: value for key, value in tag_freq.items() if key not in {'VB', 'NN', 'JJ', 'RB', 'IN', 'DT'}}
    # print(sub_tag_freq)

    # 최종적인 품사 count dictionary 생성
    final_tag_freq = {key: value for key, value in tag_freq.items() if key in {'VB', 'NN', 'JJ', 'RB', 'IN', 'DT'}}
    final_tag_freq['etc'] = sum(sub_tag_freq.values())

    return final_tag_freq

def hist_overlap(data1, data2):
    x1 = list(data1.keys())
    y1 = list(data1.values())

    plt.figure(figsize=(10, 6))
    plt.bar(x1, y1, color='green', width=0.4, label='data1', alpha=0.5)

    x2 = list(data2.keys())
    y2 = list(data2.values())
    plt.bar(x2, y2, color='blue', width=0.4, label='data1', alpha=0.5)

    plt.figure(figsize=(10, 6))
    plt.bar(x2, y2)

    plt.title('figure')
    plt.xlabel('pos')
    plt.ylabel('freq')
    plt.show()
    

# %%
# preprocess befor pos tagging
    
# Flickr8K
flickr = pd.read_csv('flickr_captions.txt', sep=',') #(image, caption) paired dataset
flickr_captions = flickr.caption
flickr_captions_list = flickr_captions.to_list()
flickr = ' '.join(flickr_captions_list)

# COCO captions
coco_captions = []
for i in range(1, 12):
    with open(f'./coco_train_captions/coco_train_{i}.json', 'r') as f:
        coco = json.load(f)
        coco_captions.extend(coco['annotations'])

coco_captions_list = [item['caption'] for item in coco_captions]
coco = ' '.join(coco_captions_list)

# %%
# bookcorpus
bk = load_dataset("bookcorpus", split='train')
print('dataset load complete')

bk =  ' '.join(bk['text'])
print('join completed')

pos_plot(bk)
# %%
# RedPajama sampled
rp_sample = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split='train')

rp_sample = ' '.join(rp_sample['text'])
pos_plot(rp_sample)


# %%
# wikipedia
wiki = load_dataset("wikipedia", "20220301.en", split="train")
print('dataset load complete')

wiki = ' '.join(wiki['text'])
print('join completed')

pos_plot(wiki)


# %%
c4 = load_dataset("c4", "en", split="validation")
print('dataset load complete')

c4 = ' '.join(c4['text'])
print('join completed')

pos_plot(c4)
# %%
