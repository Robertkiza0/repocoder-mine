"""Passe de test : dataset officiel RepoCoder + ast_chunker + retriever AST.

Contrairement à `dataset.py` (corpus reconstruit depuis prompt+right_context
de CCEval), on dispose ici des vrais fichiers sources des 8 dépôts
(`data/repos_source/`, extraits de `repocoder/repositories/line_and_api_level.zip`)
et des tâches officielles RepoCoder (`datasets rapo/`, extraites de
`repocoder/datasets/datasets.zip`). On peut donc reproduire la garde anti-fuite
de l'officiel (`search_code.py: _is_context_after_hole`) précisément, par
position de ligne réelle dans le fichier, plutôt que par exclusion de tâche
entière.
"""

import json
import os
from pathlib import Path
from typing import Any

from ast_chunker import load_and_chunk_repo_ast_cached
from retriever import retrieve_top_k_ast_jaccard

PROJECT_DIR = Path(__file__).resolve().parent
REPOS_DIR = PROJECT_DIR / "data" / "repos_source"
TASKS_PATH = PROJECT_DIR / "datasets rapo" / "line_level_completion_1k_context_codegen.test.jsonl"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def filter_safe_chunks(
    chunks: list[dict[str, Any]],
    repo_dir: Path,
    fpath_tuple: list[str],
    context_start_lineno: int,
) -> list[dict[str, Any]]:
    """Écarter les fenêtres du même fichier qui chevauchent/suivent la zone
    déjà visible dans le prompt (donc a fortiori tout ce qui est après le
    trou) — équivalent de `search_code.CodeSearchWorker._is_context_after_hole`,
    mais par position de ligne réelle puisqu'on a le vrai fichier source.
    """
    task_file = os.path.normpath(str(repo_dir.joinpath(*fpath_tuple[1:])))
    safe = []
    for chunk in chunks:
        if os.path.normpath(chunk["file_path"]) == task_file:
            chunk_end_0indexed = chunk["line_end"] - 1  # ast_chunker est 1-indexé
            if chunk_end_0indexed > context_start_lineno:
                continue
        safe.append(chunk)
    return safe


def run_demo(tasks_per_repo: int = 2) -> None:
    tasks = load_tasks(TASKS_PATH)
    print(f"{len(tasks)} tâches chargées depuis {TASKS_PATH.name}")

    seen_per_repo: dict[str, int] = {}
    chunk_cache: dict[str, list[dict[str, Any]]] = {}

    for task in tasks:
        metadata = task["metadata"]
        repo = metadata["task_id"].split("/")[0]
        if seen_per_repo.get(repo, 0) >= tasks_per_repo:
            continue
        seen_per_repo[repo] = seen_per_repo.get(repo, 0) + 1

        repo_dir = REPOS_DIR / repo
        if not repo_dir.exists():
            print(f"[SKIP] {metadata['task_id']} — dossier introuvable: {repo_dir}")
            continue

        if repo not in chunk_cache:
            print(f"\nDécoupage AST de {repo} (cache disque si déjà présent) ...")
            chunk_cache[repo] = load_and_chunk_repo_ast_cached(str(repo_dir))
            print(f"  -> {len(chunk_cache[repo])} fenêtres")

        chunks = filter_safe_chunks(
            chunk_cache[repo], repo_dir, metadata["fpath_tuple"], metadata["context_start_lineno"]
        )

        results = retrieve_top_k_ast_jaccard(task["prompt"], chunks, k=3)

        print(f"\n=== {metadata['task_id']} ({'/'.join(metadata['fpath_tuple'])}, ligne {metadata['line_no']}) ===")
        print("ground_truth:", repr(metadata["ground_truth"]))
        for rank, result in enumerate(results, start=1):
            rel_path = os.path.relpath(result["file_path"], repo_dir)
            print(f"  #{rank} score={result['score']:.3f}  {rel_path}:{result['line_start']}-{result['line_end']}")


if __name__ == "__main__":
    run_demo(tasks_per_repo=2)
