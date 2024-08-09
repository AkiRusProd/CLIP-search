from pathlib import Path
import faiss
import numpy as np

class ImageIndexer:
    def __init__(self, index_path):
        self.index = None
        self.index_path = Path(index_path)

    def fit(self, image_embeds, num_clusters=None, m=32):
        if isinstance(image_embeds, list):
            image_embeds = np.array(image_embeds)
        if num_clusters is None:
            num_clusters = min(round(np.sqrt(image_embeds.shape[0])), image_embeds.shape[0] // 39) # Emperically determined

        n_dims = image_embeds.shape[1]

        quantizer = faiss.IndexHNSWFlat(n_dims, m)
        self.index = faiss.IndexIVFFlat(quantizer, n_dims, num_clusters, faiss.METRIC_INNER_PRODUCT) # METRIC_INNER_PRODUCT: the higher this value, the better; METRIC_L2: other way round
        self.index.train(image_embeds)
        self.index.add(image_embeds)

        faiss.write_index(self.index, str(self.index_path / 'faiss_clusters.index'))

        return self.index

    def predict(self, image_embeds, embed, k):
        if isinstance(image_embeds, list):
            image_embeds = np.array(image_embeds)
        if self.index is None:
            self.index = faiss.read_index(str(self.index_path / 'faiss_clusters.index'))

        D, I = self.index.search(embed, k)
        return D[0], I[0], image_embeds[I[0]]
