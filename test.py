from src.utils.model_downloader import ModelDownloader

d = ModelDownloader()

d.download_folder(
    "sentiment/best_model/",
    "models/sentiment/best_model"
)

d.download_folder(
    "retail_ner/best_model/",
    "models/retail_ner/best_model"
)