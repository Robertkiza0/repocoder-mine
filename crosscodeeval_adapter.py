"""Adapte le format CrossCodeEval (prompt/groundtruth/right_context/metadata,
avec metadata.repository="owner-repo-sha" et metadata.file) au format de
tâche utilisé par le reste du pipeline (metadata.ground_truth/fpath_tuple/
line_no — voir les tâches officielles RepoCoder), pour réutiliser tel quel
filter_safe_chunks/build_prompt/run_condition/etc.

metadata.repository ne donne pas directement une URL GitHub clonable (c'est
"owner-repo-shortsha", ambigu à re-découper puisque owner/repo peuvent
eux-mêmes contenir des tirets) — on le résout via LICENSES/project_license_map.txt
(fourni par CCEval), qui liste les vrais "owner/repo" pour chaque projet.
"""

import json
from pathlib import Path
from typing import Any


def load_license_map(path: str | Path) -> dict[str, str]:
    """{"owner-repo": "owner/repo"} à partir de LICENSES/project_license_map.txt."""
    prefix_index: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            owner_repo, _, _ = line.partition(" ")
            if "/" not in owner_repo:
                continue
            owner, _, repo = owner_repo.partition("/")
            prefix_index[f"{owner}-{repo}"] = owner_repo
    return prefix_index


def resolve_owner_repo(repository_field: str, license_prefix_index: dict[str, str]) -> str | None:
    """"turboderp-exllama-a544085" -> "turboderp/exllama", ou None si non résolu."""
    for key, owner_repo in license_prefix_index.items():
        if repository_field.startswith(key + "-"):
            return owner_repo
    return None


def load_cceval_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def normalize_cceval_record(record: dict[str, Any], license_prefix_index: dict[str, str]) -> dict[str, Any] | None:
    """Convertit un enregistrement CCEval au format de tâche utilisé partout
    ailleurs dans le pipeline (même forme que les tâches RepoCoder). Retourne
    None si le dépôt ne peut pas être résolu vers un owner/repo GitHub."""
    metadata = record["metadata"]
    owner_repo = resolve_owner_repo(metadata["repository"], license_prefix_index)
    if owner_repo is None:
        return None

    file_parts = metadata["file"].split("/")
    fpath_tuple = [metadata["repository"]] + file_parts  # position 0 = nom de dossier du clone

    return {
        "prompt": record["prompt"],
        "metadata": {
            "task_id": metadata["task_id"],
            "ground_truth": record["groundtruth"],
            "fpath_tuple": fpath_tuple,
            "context_start_lineno": metadata["context_start_lineno"],
            "line_no": metadata["groundtruth_start_lineno"],
            "repository": metadata["repository"],
            "owner_repo": owner_repo,
        },
    }


def normalize_cceval_tasks(
    records: list[dict[str, Any]], license_prefix_index: dict[str, str]
) -> list[dict[str, Any]]:
    normalized = []
    skipped = 0
    for record in records:
        task = normalize_cceval_record(record, license_prefix_index)
        if task is None:
            skipped += 1
            continue
        normalized.append(task)
    if skipped:
        print(f"{skipped} tâche(s) ignorée(s) (dépôt non résolu vers un owner/repo GitHub)")
    return normalized


def filter_safe_chunks_cceval(chunks: list[dict[str, Any]], repo_dir, fpath_tuple: list[str]) -> list[dict[str, Any]]:
    """Comme filter_safe_chunks (RepoCoder), mais exclut ENTIÈREMENT le fichier
    de la tâche du corpus — pas seulement les lignes après le trou — car le
    dépôt cloné est la version HEAD actuelle, potentiellement différente du
    commit exact utilisé pour construire prompt/groundtruth/right_context. Se
    fier aux numéros de ligne d'un fichier qui a pu dériver serait risqué ;
    l'exclusion totale du fichier est le choix prudent."""
    import os

    task_file = os.path.normpath(str(repo_dir.joinpath(*fpath_tuple[1:])))
    return [chunk for chunk in chunks if os.path.normpath(chunk["file_path"]) != task_file]
