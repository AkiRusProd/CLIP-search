
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from clip import CLIPSearcher
from clusterer import ImageIndexer
from PIL import Image

tqdm.pandas()


class SearchMechanism:
    def __init__(self, clip_searcher: CLIPSearcher, image_indexer: ImageIndexer) -> None:
        self.clip_searcher = clip_searcher
        self.image_indexer = image_indexer

    def scan_directory(self, path):
        df = pd.DataFrame(columns=['image_path'])
        df_image_embeds = []

        for i, img in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
            df.loc[i, 'image_path'] = path + '\\' + img
            df_image_embeds.append(self.clip_searcher.get_image_features(Image.open(path + '\\' + img)).flatten())

        df.to_csv('embed_data/df.csv',  sep='\t')
        np.save('embed_data/df_image_embeds.npy', df_image_embeds)

        self.image_indexer.fit(df_image_embeds)

        return df, df_image_embeds

    def compute_similarity(self, embeds: str, df: pd.DataFrame, df_image_embeds: list, top_k: int = 5, use_cluster_search: bool = False):

        if not use_cluster_search:
            df['cos_sim'] = pd.Series(df_image_embeds).progress_apply(lambda x: torch.nn.functional.cosine_similarity(torch.tensor(x), torch.tensor(embeds)))
            df = df.sort_values(by='cos_sim', ascending=False)[: top_k]

            return df.reset_index()
        else:
            score, ids, _ = self.image_indexer.predict(df_image_embeds, embeds, top_k)
    
            ids = [id for id in ids if id != -1]
            score = score[:len(ids)]

            df = df.loc[ids]
            df['score'] = score

            df = df.sort_values(by='score', ascending=False)
            
            return df.reset_index()


    def get_top_k_text_similarities(self, text: str, df: pd.DataFrame, df_image_embeds: list, top_k: int = 5, use_cluster_search: bool = False):

        text_embeds = self.clip_searcher.get_text_features(text)
        
        return self.compute_similarity(text_embeds, df, df_image_embeds, top_k, use_cluster_search)

    def get_top_k_image_similarities(self, image, df: pd.DataFrame, df_image_embeds: list, top_k: int = 5, use_cluster_search: bool = False):

        if type(image) == str:
            image = Image.open(image)

        image_embeds = self.clip_searcher.get_image_features(image)
        
        return self.compute_similarity(image_embeds, df, df_image_embeds, top_k, use_cluster_search)
        
