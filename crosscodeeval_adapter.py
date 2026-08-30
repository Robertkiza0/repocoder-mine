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

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

_PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(_PY_LANGUAGE)


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


def _get_index(lines: list[str], point: tuple) -> int:
    return sum(len(lines[i]) for i in range(point[0])) + point[1]


def postprocess_completion(prompt: str, completion: str) -> str:
    """Coupe une complétion brute à la fin de la PREMIÈRE instruction complète,
    exactement comme le harnais officiel CCEval (cceval/metrics.py:
    _cut_first_statement_completion) — sans ça, une complétion de 64 tokens
    déborde largement au-delà du vrai point de complétion (plusieurs lignes/
    instructions en plus), et une comparaison brute (EM/ES) sous-estime
    massivement la qualité réelle de la génération.

    Si aucune instruction complète n'est trouvée (complétion tronquée en plein
    milieu), retourne la complétion telle quelle.
    """
    if len(completion.strip()) == 0:
        return completion

    # seulement les ~10 dernières lignes du prompt pour le parsing (comme l'officiel)
    line_idx = 10
    prompt_tail = "".join(prompt.splitlines(keepends=True)[-line_idx:])
    while '"""' in prompt_tail or "'''" in prompt_tail:
        line_idx -= 1
        if line_idx <= 0:
            return completion
        prompt_tail = "".join(prompt.splitlines(keepends=True)[-line_idx:])

    text = prompt_tail + completion
    lines = text.splitlines(keepends=True)
    tree = _PARSER.parse(text.encode("utf8"))
    node = tree.root_node

    while node:
        if _get_index(lines, node.start_point) > len(prompt_tail):
            return completion
        if _get_index(lines, node.end_point) <= len(prompt_tail):
            return completion
        if node.type.endswith("_statement"):
            if node.type not in [
                "for_statement", "if_statement", "with_statement",
                "while_statement", "try_statement", "match_statement",
            ]:
                break
        if node.type in ("argument_list", "parameters", "case_pattern", "attribute", "comparison_operator"):
            break
        if len(node.children) == 0:
            break
        next_node = None
        for child in node.children:
            start_index = _get_index(lines, child.start_point)
            end_index = _get_index(lines, child.end_point)
            if start_index <= len(prompt_tail) and end_index > len(prompt_tail):
                next_node = child
                break
        if next_node is None:
            break
        node = next_node

    return text[len(prompt_tail):_get_index(lines, node.end_point)]


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
