
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
        self.index_path = self.image_indexer.index_path

        self.df: pd.DataFrame = None
        self.df_image_embeds: list = None

        self.load_db()

    def load_db(self):
        try:
            if not os.path.exists(self.index_path):
                os.mkdir(self.index_path)

            self.df = pd.read_csv(os.path.join(self.index_path, 'df.csv'), sep='\t')
            self.df_image_embeds = [x.flatten() for x in np.load(os.path.join(self.index_path, 'df_image_embeds.npy'))]
        except:
            pass

    def scan_directory(self, path):
        if path is None or not os.path.exists(path):
            raise Exception("Path does not exist")

        df = pd.DataFrame(columns=['image_path'])
        df_image_embeds = []

        for i, img in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
            df.loc[i, 'image_path'] = path + '\\' + img
            df_image_embeds.append(self.clip_searcher.get_image_features(Image.open(path + '\\' + img)).flatten())

        df.to_csv(os.path.join(self.index_path, 'df.csv'),  sep='\t')
        np.save(os.path.join(self.index_path, 'df_image_embeds.npy'), df_image_embeds)

        self.image_indexer.fit(df_image_embeds)

        self.load_db()

    def query_by_embeds(self, embeds: str, top_k: int = 5, use_cluster_search: bool = False):
        if self.df is None or self.df_image_embeds is None:
            return

        df = self.df

        if not use_cluster_search:
            df['cos_sim'] = pd.Series(self.df_image_embeds).progress_apply(lambda x: torch.nn.functional.cosine_similarity(torch.tensor(x), torch.tensor(embeds)))
            df = df.sort_values(by='cos_sim', ascending=False)[: top_k]

            return df.reset_index()
        else:
            score, ids, _ = self.image_indexer.predict(self.df_image_embeds, embeds, top_k)
    
            ids = [id for id in ids if id != -1]
            score = score[:len(ids)]

            df = df.loc[ids]
            df['score'] = score

            df = df.sort_values(by='score', ascending=False)
            
            return df.reset_index()


    def query_by_text(self, text: str, top_k: int = 5, use_cluster_search: bool = False):
        text_embeds = self.clip_searcher.get_text_features(text)
        
        return self.query_by_embeds(text_embeds, top_k, use_cluster_search)

    def query_by_image(self, image, top_k: int = 5, use_cluster_search: bool = False):
        if type(image) == str:
            image = Image.open(image)

        image_embeds = self.clip_searcher.get_image_features(image)
        
        return self.query_by_embeds(image_embeds, top_k, use_cluster_search)
        
