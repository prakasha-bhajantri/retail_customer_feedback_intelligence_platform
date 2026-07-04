from pathlib import Path
import shutil

from google.cloud import storage

from src.core.config import config


class ModelDownloader:

    def __init__(self):

        self.project_id = config["gcp"]["project_id"]
        self.bucket_name = config["storage"]["bucket"]

        self.client = storage.Client(
            project=self.project_id
        )

    def download_folder(self, prefix: str, destination: str):

        destination = Path(destination)

        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ]

        if all((destination / f).exists() for f in required_files):
            print(f"{destination} already exists.")
            return

        if destination.exists():
            shutil.rmtree(destination)

        destination.mkdir(parents=True, exist_ok=True)

        try:

            blobs = self.client.list_blobs(
                self.bucket_name,
                prefix=prefix,
            )

            for blob in blobs:

                if blob.name.endswith("/"):
                    continue

                local_file = destination / Path(blob.name).relative_to(prefix)

                local_file.parent.mkdir(parents=True, exist_ok=True)

                print(f"Downloading {blob.name}")

                blob.download_to_filename(local_file)

        except Exception as e:
            raise RuntimeError(
                f"Failed to download model '{prefix}' from bucket '{self.bucket_name}'"
            ) from e

        print(f"{prefix} download complete.")

    def download_all_models(self):

        self.download_folder(
            "sentiment/best_model/",
            "models/sentiment/best_model"
        )

        self.download_folder(
            "retail_ner/best_model/",
            "models/retail_ner/best_model"
        )        