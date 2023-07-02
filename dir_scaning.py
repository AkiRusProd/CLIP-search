
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from clip import clip_searcher
from clusterer import image_indexer
from PIL import Image


tqdm.pandas()

os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:\\Code\\Huggingface_cache\\'

path = 'D:\\Code\\Diffusion_Images'

def scan_directory(path):
    df = pd.DataFrame(columns=['image_path'])
    df_image_embeds = []

    for i, img in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
        df.loc[i, 'image_path'] = path + '\\' + img
        df_image_embeds.append(clip_searcher.get_image_features(Image.open(path + '\\' + img)).flatten())

    df.to_csv('embed_data/df.csv',  sep='\t')
    np.save('embed_data/df_image_embeds.npy', df_image_embeds)

    image_indexer.fit(df_image_embeds)

    return df, df_image_embeds

def compute_similarity(embeds: str, df: pd.DataFrame, df_image_embeds: list, top_k: int = 5, use_cluster_search: bool = False):

    if not use_cluster_search:
        df['cos_sim'] = pd.Series(df_image_embeds).progress_apply(lambda x: torch.nn.functional.cosine_similarity(torch.tensor(x), torch.tensor(embeds)))
        df = df.sort_values(by='cos_sim', ascending=False)[: top_k]

        return df.reset_index()
    else:
        score, ids, _ = image_indexer.predict(df_image_embeds, embeds, top_k)
  
        ids = [id for id in ids if id != -1]
        score = score[:len(ids)]

        df = df.loc[ids]
        df['score'] = score

        df = df.sort_values(by='score', ascending=False)
        
        return df.reset_index()


def get_top_k_text_similarities(text: str, df: pd.DataFrame, df_image_embeds: list, top_k: int = 5, use_cluster_search: bool = False):

    text_embeds = clip_searcher.get_text_features(text)
    
    return compute_similarity(text_embeds, df, df_image_embeds, top_k, use_cluster_search)

def get_top_k_image_similarities(image, df: pd.DataFrame, df_image_embeds: list, top_k: int = 5, use_cluster_search: bool = False):

    if type(image) == str:
        image = Image.open(image)

    image_embeds = clip_searcher.get_image_features(image)
    
    return compute_similarity(image_embeds, df, df_image_embeds, top_k, use_cluster_search)
        

# scan_directory(path)

# df = pd.read_csv('df.csv', sep='\t')
# df_image_embeds = np.load('embed_data/df_image_embeds.npy')
# df_image_embeds = [x.flatten() for x in df_image_embeds]


# top_k_df = get_top_k_text_similarities("guy with smartphone in hand", df, df_image_embeds)[['image_path', 'cos_sim']]
# top_k_df.to_csv('top_k.csv',  sep='\t')

# top_k_df = get_top_k_image_similarities("D:\\Code\Diffusion_Images\\2023-05-11 17-43-18.281968.jpg", df, df_image_embeds)[['image_path', 'cos_sim']]
# top_k_df.to_csv('top_k.csv',  sep='\t')