from logging import FileHandler as loggingFileHandler
from pathlib import Path
from typing import Any


class FileHandler(loggingFileHandler):
    """
    A handler class which writes formatted logging records to disk files.

    Wrapper of `logging.FileHandler` that creates parent directory of the log
    file automatically.
    """

    def __init__(self, filename: str | Path, *args: Any, **kwargs: Any):
        # ensure existence of parent directory
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(filename, *args, **kwargs)
