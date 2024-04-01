
import torch
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from dotenv import dotenv_values

env = dotenv_values('.env')

# your models cache will be stored here
os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']

model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"




class CLIPSearcher:
    def __init__(self, model_id, device):
        self.model: CLIPModel = CLIPModel.from_pretrained(model_id).to(device)
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_id)

    def get_text_features(self, text):
        inputs = self.tokenizer(text, return_tensors = "pt").to(device)
        return self.model.get_text_features(**inputs).cpu().detach().numpy()

    def get_image_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        return self.model.get_image_features(**inputs).cpu().detach().numpy()


clip_searcher = CLIPSearcher(model_id, device)



