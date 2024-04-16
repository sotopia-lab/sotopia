import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch


# IO
def load_data(text_path: str | os.PathLike[str], encoding: str = "utf-8") -> list[str]:
    """
    load textual data from file
    """
    with open(text_path, encoding=encoding) as fp:
        texts = fp.readlines()
    return [t.strip() for t in texts]


def save_data(filename: str, data: str) -> None:
    """
    write textual data to file
    """
    with open(filename, "w") as fout:
        for d in data:
            fout.write(str(d) + "\n")
    fout.close()


def save_jsonl(path: str, entries: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf8") as fh:
        for entry in entries:
            fh.write(f"{json.dumps(entry)}\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    pairs = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            pairs.append(json.loads(line))
    return pairs


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf8") as fh:
        content: dict[str, Any] = json.load(fh)
    return content


def save_json(path: str, content: Any) -> None:
    with open(path, "w", encoding="utf8") as fh:
        json.dump(content, fh, indent=4)


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
