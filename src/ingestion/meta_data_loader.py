from src.ingestion.base_loader import BaseLoader


class MetadataLoader(BaseLoader):

    def __init__(self, file_path):

        super().__init__(
            file_path=file_path,
            parser="literal"
        )