import os
import gradio as gr

from PIL import Image
from db import SearchMechanism
from clip import CLIPSearcher
from clusterer import ImageIndexer
from dotenv import dotenv_values

env = dotenv_values('.env')

path = env['DEFAULT_IMAGES_PATH']
index_path = env['INDEX_PATH']

# your models cache will be stored here
os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']


def search_by_text(text, top_k, use_cluster_search):
    top_k_df = search_mechanism.query_by_text(text, top_k, use_cluster_search)
    images = [Image.open(row['image_path']) for _, row in top_k_df.iterrows()]

    return images

def search_by_image(image, top_k, use_cluster_search):
    top_k_df = search_mechanism.query_by_image(image, top_k, use_cluster_search)
    images = [Image.open(row['image_path']) for _, row in top_k_df.iterrows()]

    return images

def scan_dir(path):
    if path is None or not os.path.exists(path):
        return gr.Info("Path does not exist")

    search_mechanism.scan_directory(path)

    return path

clip_searcher = CLIPSearcher()
image_indexer = ImageIndexer(index_path)
search_mechanism = SearchMechanism(clip_searcher, image_indexer)

with gr.Blocks() as webui:
    gr.Markdown("CLIP Searcher")

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label = "Text", info = "Text to search")

            with gr.Row():
                image = gr.Image(label = "Image")

                with gr.Column():
                    top_k_slider = gr.Slider(label="Top K", minimum=1, maximum=50, step=1, value=5, info = "Top K closest results to the query")
                    use_cluster_search = gr.Checkbox(label="Use cluster search", value=False, info="Faster embedding search using clusters, may be less accurate")
                    search_by_text_btn = gr.Button("Search by text")
                    search_by_image_btn = gr.Button("Search by image")

            path = gr.Textbox(label = "Path", info = "Path with images to scan", value=path)
            scan_dir_btn = gr.Button("Scan directory", variant="primary")

  
        gallery = gr.Gallery(label = "Gallery", show_label=False, columns=3, rows = 2, height="auto", preview = False)

    search_by_text_btn.click(
        search_by_text, 
        inputs = [text, top_k_slider, use_cluster_search], 
        outputs = gallery
    )

    search_by_image_btn.click(
        search_by_image,
        inputs = [image, top_k_slider, use_cluster_search],
        outputs = gallery
    )

    scan_dir_btn.click(
        scan_dir,
        inputs = [path],
        outputs = path
    )

webui.queue()
webui.launch()