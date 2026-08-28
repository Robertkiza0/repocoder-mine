"""CrossCodeEval JSONL 데이터셋 로딩."""

import logging
import os
import random
from typing import Literal

from .base import Example

Split = Literal[
    "baseline",
    "bm25",
    "unixcoder",
    "openai",
    "oracle_bm25",
    "oracle_unixcoder",
    "oracle_openai",
]

logger = logging.getLogger(__name__)


def load_cceval_dataset(
    path: str | None = None,
    language: str = "python",
    split: Split = "baseline",
    sample: int | None = None,
    seed: int = 42,
) -> list[Example]:
    """CCEval JSONL 데이터셋을 로드하여 Example 리스트로 반환.

    path를 주지 않으면 CCEVAL_DATA_DIR 환경변수를 루트로 하여
    `<CCEVAL_DATA_DIR>/<language>/<filename>` 경로를 추정한다.
    이 경우 CCEVAL_DATA_DIR가 설정되어 있지 않으면 에러가 발생한다.
    """
    if path is None:
        path = _resolve_cceval_path(split, language)

    logger.info("Loading dataset: %s", path)
    with open(path) as f:
        items = [Example.model_validate_json(line) for line in f]

    keys = {item.metadata.task_id for item in items}
    assert len(items) == len(keys), "Duplicate key found"

    if sample is not None:
        random.seed(seed)
        items = random.sample(items, min(sample, len(items)))

    logger.info("Loaded %d records", len(items))
    return items


def _resolve_cceval_path(split: str = "baseline", language: str = "python") -> str:
    cceval_data_dir = os.getenv("CCEVAL_DATA_DIR")
    if cceval_data_dir is None:
        raise ValueError("CCEVAL_DATA_DIR environment variable is not set")
    if split == "baseline":
        filename = "line_completion.jsonl"
    elif split == "bm25":
        filename = "line_completion_rg1_bm25.jsonl"
    elif split == "unixcoder":
        filename = "line_completion_rg1_unixcoder_cosine_sim.jsonl"
    elif split == "openai":
        filename = "line_completion_rg1_openai_cosine_sim.jsonl"
    elif split == "oracle_bm25":
        filename = "line_completion_oracle_bm25.jsonl"
    elif split == "oracle_unixcoder":
        filename = "line_completion_oracle_unixcoder_cosine_sim.jsonl"
    elif split == "oracle_openai":
        filename = "line_completion_oracle_openai_cosine_sim.jsonl"
    else:
        raise ValueError(f"Invalid split: {split}")
    return os.path.join(cceval_data_dir, language, filename)
