from pathlib import Path
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
        self.index_path = Path(self.image_indexer.index_path)

        self.df: pd.DataFrame = None
        self.df_image_embeds: list = None

        self.load_db()

    def load_db(self):
        try:
            if not self.index_path.exists():
                self.index_path.mkdir(parents=True, exist_ok=True)

            df_path = self.index_path / 'df.csv'
            embeds_path = self.index_path / 'df_image_embeds.npy'

            if df_path.exists() and embeds_path.exists():
                self.df = pd.read_csv(df_path, sep='\t')
                self.df_image_embeds = [x.flatten() for x in np.load(embeds_path)]
        except Exception as e:
            print(f"Error loading database: {e}")

    def scan_directory(self, path: Path):
        if path is None or not path.exists():
            raise Exception("Path does not exist")

        df = pd.DataFrame(columns=['image_path'])
        df_image_embeds = []

        image_files = list(path.iterdir())
        for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
            df.loc[i, 'image_path'] = str(img_path)
            df_image_embeds.append(self.clip_searcher.get_image_features(Image.open(img_path)).flatten())

        df.to_csv(self.index_path / 'df.csv', sep='\t', index=False)
        np.save(self.index_path / 'df_image_embeds.npy', df_image_embeds)

        self.image_indexer.fit(df_image_embeds)

        self.load_db()

    def query_by_embeds(self, embeds: np.ndarray, top_k: int = 5, use_cluster_search: bool = False):
        if self.df is None or self.df_image_embeds is None:
            return

        df = self.df

        if not use_cluster_search:
            df['cos_sim'] = pd.Series(self.df_image_embeds).progress_apply(
                lambda x: torch.nn.functional.cosine_similarity(torch.tensor(x), torch.tensor(embeds))
            )
            df = df.sort_values(by='cos_sim', ascending=False).head(top_k)

            return df.reset_index(drop=True)
        else:
            score, ids, _ = self.image_indexer.predict(self.df_image_embeds, embeds, top_k)

            ids = [id for id in ids if id != -1]
            score = score[:len(ids)]

            df = df.iloc[ids].copy()
            df['score'] = score

            df = df.sort_values(by='score', ascending=False)
            
            return df.reset_index(drop=True)

    def query_by_text(self, text: str, top_k: int = 5, use_cluster_search: bool = False):
        text_embeds = self.clip_searcher.get_text_features(text)
        
        return self.query_by_embeds(text_embeds, top_k, use_cluster_search)

    def query_by_image(self, image, top_k: int = 5, use_cluster_search: bool = False):
        if isinstance(image, str):
            image = Image.open(image)

        image_embeds = self.clip_searcher.get_image_features(image)
        
        return self.query_by_embeds(image_embeds, top_k, use_cluster_search)
