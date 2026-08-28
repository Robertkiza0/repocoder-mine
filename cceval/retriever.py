import json
import logging
from pathlib import Path

from .base import CodeChunk
from .dataset import load_cceval_dataset

logger = logging.getLogger(__name__)


KEYWORD_DATASET: dict[str, dict[str, list[CodeChunk]]] = {}


def _get_keyword_dataset(lang: str, keyword_dir: Path) -> dict[str, list[CodeChunk]]:
    if lang in KEYWORD_DATASET:
        return KEYWORD_DATASET[lang]
    dataset_path = keyword_dir / lang / "retrieved.jsonl"
    dataset = {}
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            dataset[item["task_id"]] = [
                CodeChunk.model_validate(chunk) for chunk in item["retrieved_chunks"]
            ]
    KEYWORD_DATASET[lang] = dataset
    return KEYWORD_DATASET[lang]


DATAFLOW_DATASET: dict[tuple[str, str], dict[str, list[CodeChunk]]] = {}


def _get_dataflow_dataset(
    lang: str, model_name: str, dataflow_dir: Path
) -> dict[str, list[CodeChunk]]:
    if (lang, model_name) in DATAFLOW_DATASET:
        return DATAFLOW_DATASET[(lang, model_name)]

    MODEL_MAP = {
        "bigcode/starcoder": "starcoder",
        "bigcode/starcoderbase-7b": "starcoderbase",
        "bigcode/starcoder2-7b": "starcoder2",
        "Qwen/Qwen2.5-Coder-7B": "qwen25coder",
        "infly/OpenCoder-8B-Base": "opencoder",
        "01-ai/Yi-Coder-9B": "yicoder",
    }
    dataset_path = (
        dataflow_dir / MODEL_MAP[model_name] / "line_completion_rg1_dataflow.jsonl"
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataflow dataset file not found: {dataset_path}")

    dataset = load_cceval_dataset(dataset_path)
    dataset = {item.metadata.task_id: item.crossfile_context.list for item in dataset}
    DATAFLOW_DATASET[(lang, model_name)] = dataset
    return dataset


DATASET: dict[tuple[str, str, str | None, bool], dict[str, list[CodeChunk]]] = {}


def _get_cceval_dataset(
    lang: str, method: str, with_gt: bool, crosscodeeval_dir: Path
) -> dict[str, list[CodeChunk]]:
    """주어진 method에 대해 전체 데이터셋을 반환합니다."""
    if (lang, method, with_gt) in DATASET:
        return DATASET[(lang, method, with_gt)]

    if method == "bm25":
        if with_gt:
            filename = "line_completion_oracle_bm25.jsonl"
        else:
            filename = "line_completion_rg1_bm25.jsonl"
    elif method == "dense":
        if with_gt:
            filename = "line_completion_oracle_openai_cosine_sim.jsonl"
        else:
            filename = "line_completion_rg1_openai_cosine_sim.jsonl"
    else:
        raise NotImplementedError(f"Unknown method: {method}")
    dataset_path = crosscodeeval_dir / lang / filename

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_cceval_dataset(dataset_path)
    dataset = {item.metadata.task_id: item.crossfile_context.list for item in dataset}
    DATASET[(lang, method, with_gt)] = dataset
    return dataset


def get_dataflow_retrieved_chunk(
    task_id: str,
    lang: str,
    model_name: str,
    dataflow_dir: Path,
) -> list[CodeChunk]:
    """Dataflow method에 대해 주어진 task_id에 해당하는 CodeChunk 리스트를 반환합니다."""
    dataset = _get_dataflow_dataset(lang, model_name, dataflow_dir)
    if task_id not in dataset:
        raise KeyError(
            f"Task ID {task_id} not found in dataflow dataset for language {lang} and model {model_name}"
        )
    return dataset[task_id]


def get_keyword_retrieved_chunk(
    task_id: str,
    lang: str,
    keyword_dir: Path,
) -> list[CodeChunk]:
    """Keyword method에 대해 주어진 task_id에 해당하는 CodeChunk 리스트를 반환합니다."""
    dataset = _get_keyword_dataset(lang, keyword_dir)
    if task_id not in dataset:
        raise KeyError(
            f"Task ID {task_id} not found in keyword dataset for language {lang}"
        )
    return dataset[task_id]


def get_cceval_retrieved_chunk(
    task_id: str,
    lang: str,
    method: str,
    crosscodeeval_dir: Path,
    with_gt: bool,
) -> list[CodeChunk]:
    """BM25, Dense method에 대해 주어진 task_id에 해당하는 CodeChunk 리스트를 반환합니다."""
    dataset = _get_cceval_dataset(lang, method, crosscodeeval_dir, with_gt=with_gt)
    if task_id not in dataset:
        raise KeyError(
            f"Task ID {task_id} not found in dataset for language {lang}, method {method}, with_gt {with_gt}"
        )
    return dataset[task_id]
