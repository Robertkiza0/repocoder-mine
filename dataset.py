import json
import os
import random
import re
from pathlib import Path
from typing import Any, Literal


REQUIRED_FIELDS = {"prompt", "groundtruth", "right_context"}
Split = Literal[
    "baseline",
    "bm25",
    "unixcoder",
    "openai",
    "oracle_bm25",
    "oracle_unixcoder",
    "oracle_openai",
]


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Charger les exemples valides d'un fichier JSONL."""
    records = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Ligne ignorée {line_number}: JSON invalide ({error})")
                continue

            if not isinstance(record, dict):
                print(f"Ligne ignorée {line_number}: l'objet n'est pas un dictionnaire")
                continue

            missing_fields = REQUIRED_FIELDS - record.keys()
            if missing_fields:
                print(
                    f"Ligne ignorée {line_number}: champs manquants "
                    f"{sorted(missing_fields)}"
                )
                continue

            if any(not isinstance(record[field], str) for field in REQUIRED_FIELDS):
                print(f"Ligne ignorée {line_number}: un champ contient une valeur invalide")
                continue

            records.append(record)

    return records


def load_cceval_examples(
    path: str | Path | None = None,
    language: str = "python",
    split: Split = "baseline",
    sample: int | None = None,
    seed: int = 42,
):
    """Charger les exemples avec le modèle officiel de CrossCodeEval."""
    from cceval.dataset import load_cceval_dataset as load_official_dataset

    return load_official_dataset(
        path=str(path) if path is not None else None,
        language=language,
        split=split,
        sample=sample,
        seed=seed,
    )


def load_cceval_dataset(
    path: str | Path | None = None,
    language: str = "python",
    split: Split = "baseline",
    sample: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Charger le dataset CrossCodeEval avec ses champs d'évaluation."""
    if path is None:
        path = resolve_cceval_path(split, language)

    records = load_jsonl(path)
    task_ids = [record.get("metadata", {}).get("task_id") for record in records]
    task_ids = [task_id for task_id in task_ids if task_id is not None]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Le dataset contient des task_id en double")

    if sample is not None:
        if sample < 0:
            raise ValueError("sample doit être positif")
        records = random.Random(seed).sample(records, min(sample, len(records)))

    return records


def resolve_cceval_path(split: Split = "baseline", language: str = "python") -> Path:
    """Construire le chemin CrossCodeEval depuis CCEVAL_DATA_DIR."""
    data_dir = os.environ.get("CCEVAL_DATA_DIR")
    if data_dir is None:
        raise ValueError("La variable CCEVAL_DATA_DIR n'est pas définie")

    filenames = {
        "baseline": "line_completion.jsonl",
        "bm25": "line_completion_rg1_bm25.jsonl",
        "unixcoder": "line_completion_rg1_unixcoder_cosine_sim.jsonl",
        "openai": "line_completion_rg1_openai_cosine_sim.jsonl",
        "oracle_bm25": "line_completion_oracle_bm25.jsonl",
        "oracle_unixcoder": "line_completion_oracle_unixcoder_cosine_sim.jsonl",
        "oracle_openai": "line_completion_oracle_openai_cosine_sim.jsonl",
    }
    return Path(data_dir) / language / filenames[split]


SLIDING_WINDOW_SIZE = 20  # S_w dans l'article RepoCoder
SLIDING_STRIDE = 10  # S_s dans l'article RepoCoder


def slide_over_text(
    text: str,
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
) -> list[str]:
    """Découper un texte en fenêtres glissantes de lignes (S_w, S_s de RepoCoder)."""
    if window_size <= 0:
        raise ValueError("window_size doit être supérieur à 0")
    if stride <= 0 or stride > window_size:
        raise ValueError("stride doit être compris entre 1 et window_size")

    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    windows = []
    for start in range(0, len(lines), stride):
        window_text = "".join(lines[start:start + window_size]).strip()
        if window_text:
            windows.append(window_text)
        if start + window_size >= len(lines):
            break
    return windows


def last_lines(text: str, n: int = SLIDING_STRIDE) -> str:
    """Garder les n dernières lignes d'un texte (utilisé comme requête de retrieval)."""
    lines = text.splitlines(keepends=True)
    return "".join(lines[-n:])


def first_lines(text: str, n: int = SLIDING_STRIDE) -> str:
    """Garder les n premières lignes d'un texte (utilisé sur la prédiction précédente)."""
    lines = text.splitlines(keepends=True)
    return "".join(lines[:n])


def extract_repository_snippets(
    records: list[dict[str, Any]],
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
) -> dict[str, list[dict[str, Any]]]:
    """Regrouper les exemples CCEval par dépôt puis extraire leurs fenêtres glissantes.

    Reproduit le découpage de RepoCoder (S_w=20, S_s=10 par défaut) : chaque
    dépôt (metadata.repository) reçoit la liste des morceaux de code obtenus
    en faisant glisser une fenêtre sur les lignes de chaque exemple.
    """
    repositories: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        metadata = record.get("metadata", {})
        repository = metadata.get("repository") if isinstance(metadata, dict) else None
        if not repository:
            continue

        file_content = record["prompt"] + record.get("right_context", "")
        snippets = slide_over_text(file_content, window_size=window_size, stride=stride)

        for snippet in snippets:
            repositories.setdefault(repository, []).append(
                {
                    "task_id": metadata.get("task_id"),
                    "file": metadata.get("file"),
                    "snippet": snippet,
                }
            )

    return repositories


def extract_cceval_repository_snippets(
    path: str | Path | None = None,
    language: str = "python",
    split: Split = "baseline",
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
    sample: int | None = None,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Charger le dataset CCEval puis extraire les fenêtres glissantes par dépôt.

    Utilise par défaut les hyperparamètres de l'article RepoCoder
    (S_w=20, S_s=10) pour construire la base de code de chaque dépôt.
    """
    records = load_cceval_dataset(
        path=path, language=language, split=split, sample=sample, seed=seed
    )
    return extract_repository_snippets(records, window_size=window_size, stride=stride)


def save_repository_snippets(
    repositories: dict[str, list[dict[str, Any]]], output_dir: str | Path
) -> dict[str, int]:
    """Sauvegarder les fenêtres glissantes de chaque dépôt dans son propre fichier JSONL."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counts = {}
    for repository, snippets in repositories.items():
        safe_name = re.sub(r"[^\w.-]", "_", repository)
        save_jsonl(snippets, output_path / f"{safe_name}.jsonl")
        counts[repository] = len(snippets)

    return counts


def prepare_repository_snippets(
    output_dir: str | Path,
    path: str | Path | None = None,
    language: str = "python",
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
) -> dict[str, int]:
    """Extraire les fenêtres glissantes du split baseline (line_completion.jsonl) et les sauvegarder par dépôt."""
    repositories = extract_cceval_repository_snippets(
        path=path,
        language=language,
        split="baseline",
        window_size=window_size,
        stride=stride,
    )
    return save_repository_snippets(repositories, output_dir)


def split_dataset(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Mélanger les exemples puis créer les ensembles train/validation/test."""
    if train_ratio <= 0 or validation_ratio < 0:
        raise ValueError("Les ratios doivent être positifs")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("La somme des ratios doit être inférieure à 1")

    shuffled_records = records.copy()
    random.Random(seed).shuffle(shuffled_records)

    train_end = int(len(shuffled_records) * train_ratio)
    validation_end = train_end + int(len(shuffled_records) * validation_ratio)

    return {
        "train": shuffled_records[:train_end],
        "validation": shuffled_records[train_end:validation_end],
        "test": shuffled_records[validation_end:],
    }


def split_by_repository(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Séparer les dépôts entiers pour éviter une fuite entre les ensembles."""
    repositories = {}
    records_without_repository = []

    for record in records:
        metadata = record.get("metadata", {})
        repository = metadata.get("repository") if isinstance(metadata, dict) else None
        if repository:
            repositories.setdefault(repository, []).append(record)
        else:
            records_without_repository.append(record)

    if not repositories:
        return split_dataset(records, train_ratio, validation_ratio, seed)

    repository_names = list(repositories)
    random.Random(seed).shuffle(repository_names)
    train_repository_end = max(1, int(len(repository_names) * train_ratio))
    validation_repository_end = train_repository_end + int(
        len(repository_names) * validation_ratio
    )

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for repository in repository_names[:train_repository_end]:
        splits["train"].extend(repositories[repository])
    for repository in repository_names[train_repository_end:validation_repository_end]:
        splits["validation"].extend(repositories[repository])
    for repository in repository_names[validation_repository_end:]:
        splits["test"].extend(repositories[repository])

    # Les exemples sans dépôt sont répartis uniquement après le découpage principal.
    splits["train"].extend(records_without_repository)
    return splits


def save_jsonl(records: list[dict[str, Any]], file_path: str | Path) -> None:
    """Sauvegarder une liste d'exemples au format JSONL."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(input_path: str | Path, output_dir: str | Path) -> dict[str, int]:
    """Charger, séparer et sauvegarder le dataset dans trois fichiers."""
    records = load_jsonl(input_path)
    splits = split_by_repository(records)
    output_path = Path(output_dir)

    for split_name, split_records in splits.items():
        save_jsonl(split_records, output_path / f"{split_name}.jsonl")

    return {split_name: len(split_records) for split_name, split_records in splits.items()}


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    source = Path(r"C:\Users\User\Downloads\line_completion.jsonl")
    destination = project_dir / "data"
    counts = prepare_dataset(source, destination)

    print("Dataset organisé :")
    for split_name, count in counts.items():
        print(f"- {split_name}: {count} exemples")

    # Première extraction : fenêtres glissantes (S_w=20, S_s=10) par dépôt,
    # à partir de line_completion.jsonl uniquement, sauvegardées dans un dossier dédié.
    repository_destination = project_dir / "data" / "repositories"
    repository_counts = prepare_repository_snippets(repository_destination, path=source)

    print(f"\nFenêtres glissantes sauvegardées dans {repository_destination} :")
    for repository, count in repository_counts.items():
        print(f"- {repository}: {count} fenêtres")
