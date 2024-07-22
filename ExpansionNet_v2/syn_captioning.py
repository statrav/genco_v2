"""
    Small script for testing on few generic images given the model weights.
    In order to minimize the requirements, it runs only on CPU and images are
    processed one by one.
"""
#%%
import torch
import argparse
import pickle
import json
import os
import re
from tqdm import tqdm
import pandas as pd
from argparse import Namespace

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description

coco_full = []
for i in range(1, 12):
    with open(f'../data/coco_train_captions/coco_train_{i}.json', 'r') as f:
        coco = json.load(f)
        coco_full.extend(coco['annotations'])
coco_full = {'annotations': coco_full}
dataset = coco_full.get('annotations', [])

#%%

drop_args = Namespace(enc=0.0,
                        dec=0.0,
                        enc_input=0.0,
                        dec_input=0.0,
                        other=0.0)
model_args = Namespace(model_dim=512,
                        N_enc=3,
                        N_dec=3,
                        drop_args=drop_args)

with open('./demo_material/demo_coco_tokens.pickle', 'rb') as f:
    coco_tokens = pickle.load(f)
    sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
    eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

print("Dictionary loaded ...")

img_size = 384
model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                            swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                            swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                            swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                            swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                            swin_use_checkpoint=False,
                            final_swin_dim=1536,

                            d_model=model_args.model_dim, N_enc=model_args.N_enc,
                            N_dec=model_args.N_dec, num_heads=8, ff=2048,
                            num_exp_enc_list=[32, 64, 128, 256, 512],
                            num_exp_dec=16,
                            output_word2idx=coco_tokens['word2idx_dict'],
                            output_idx2word=coco_tokens['idx2word_list'],
                            max_seq_len=74, drop_args=model_args.drop_args,
                            rank='cpu')
checkpoint = torch.load('download/rf_model-002.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded ...")

input_images = []

coco_path = '../data/coco_train_images/train2017'

image_paths = [os.path.join(coco_path, file_name) for file_name in os.listdir(coco_path)[100000:]]

for path in image_paths:
    input_images.append(preprocess_image(path, img_size))

print("Generating captions ...\n")

result = []

for i in tqdm(range(len(input_images))):
    path = image_paths[i]
    image = input_images[i]
    beam_search_kwargs = {'beam_size': 5,
                            'beam_max_seq_len': 74,
                            'sample_or_max': 'max',
                            'how_many_outputs': 1,
                            'sos_idx': sos_idx,
                            'eos_idx': eos_idx}
    with torch.no_grad():
        pred, _ = model(enc_x=image,
                        enc_x_num_pads=[0],
                        mode='beam_search', **beam_search_kwargs)
    pred = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)
    index = path[-16:]
    index_list = [re.match(r'^0*([1-9][0-9]*)', index).group(1)]
    for x in dataset:
        if str(x['image_id']) == str(index_list):
            result_list = x['caption']
            break
        else:
            result_list = "None"
    
    result_data = {'path': index, 'captioning': pred, 'gt': result_list}
    result.append(result_data)

result_df = pd.DataFrame(result)
result_df.to_csv('./output/captioning_coco_11.csv')

print("Closed.")

# %%
