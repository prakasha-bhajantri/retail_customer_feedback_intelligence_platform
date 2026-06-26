# src/ingestion/base_loader.py

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
from typing import Dict, Generator


class BaseLoader:

    def __init__(self, file_path: str | Path, parser: str = "json"):

        self.file_path = Path(file_path)
        self.parser = parser

    def _parse(self, line: str) -> Dict:

        if self.parser == "json":
            return json.loads(line)

        if self.parser == "literal":
            return ast.literal_eval(line)

        raise ValueError(f"Unknown parser: {self.parser}")

    def stream(self) -> Generator[Dict, None, None]:

        with gzip.open(self.file_path, "rt", encoding="utf-8") as f:

            for line in f:

                if line.strip():

                    yield self._parse(line)