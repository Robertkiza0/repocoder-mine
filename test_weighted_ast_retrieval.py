"""Passe de test du retriever "Jaccard pondéré + attention AST" sur les 50
tâches officielles RepoCoder (mêmes données que test_ast_retrieval.py).

Construit ce qui manquait pour appeler weighted_ast_attention_score sur de
vraies données :
- query_vars (avec attention) via ast_distance.compute_query_vars_with_attention
  sur le VRAI fichier complet (pas le prompt tronqué, souvent syntaxiquement
  invalide isolément).
- doc_weights (IDF) calculés par dépôt, sur l'ensemble des fenêtres AST déjà
  produites par ast_chunker.
- chunk_symbols / chunk_imports séparés à partir de chunk['identifiers']
  (fusionnés par ast_chunker) via l'ensemble des noms d'import connus du
  dépôt — approximation : un nom traité comme import partout où il apparaît
  dans le dépôt, pas fichier par fichier.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from ast_chunker import load_and_chunk_repo_ast_cached, build_scope_map
from ast_distance import compute_query_vars_with_attention
from weighted_ast_scorer import weighted_ast_attention_score

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
    chunks: list[dict[str, Any]], repo_dir: Path, fpath_tuple: list[str], context_start_lineno: int
) -> list[dict[str, Any]]:
    task_file = os.path.normpath(str(repo_dir.joinpath(*fpath_tuple[1:])))
    safe = []
    for chunk in chunks:
        if os.path.normpath(chunk["file_path"]) == task_file:
            if chunk["line_end"] - 1 > context_start_lineno:
                continue
        safe.append(chunk)
    return safe


def compute_doc_weights(chunks: list[dict[str, Any]]) -> dict[str, float]:
    """IDF par dépôt : doc_weights[symbole] = log((N+1)/(df+1)) + 1, N = nb de
    fenêtres du dépôt, df = nb de fenêtres contenant ce symbole."""
    n_docs = len(chunks)
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update(set(chunk["identifiers"]))
    return {symbol: math.log((n_docs + 1) / (count + 1)) + 1 for symbol, count in df.items()}


def compute_repo_import_names(repo_dir: Path) -> set[str]:
    """Union des noms importés (haut niveau) dans tous les .py du dépôt —
    approximation utilisée pour séparer chunk_symbols/chunk_imports à partir
    de chunk['identifiers'] (fusionnés par ast_chunker)."""
    import_names: set[str] = set()
    for root, _, files in os.walk(repo_dir):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            file_path = os.path.join(root, filename)
            try:
                code = open(file_path, "r", encoding="utf-8").read()
                module_imports, _ = build_scope_map(code)
            except (SyntaxError, OSError):
                continue
            import_names |= module_imports
    return import_names


def split_chunk_symbols(chunk: dict[str, Any], repo_import_names: set[str]) -> tuple[set[str], set[str]]:
    identifiers = set(chunk["identifiers"])
    chunk_imports = identifiers & repo_import_names
    chunk_symbols = identifiers - chunk_imports
    return chunk_symbols, chunk_imports


def retrieve_top_k_weighted(
    query_vars: dict[str, float],
    query_imports: set[str],
    chunks: list[dict[str, Any]],
    repo_import_names: set[str],
    doc_weights: dict[str, float],
    k: int = 10,
) -> list[dict[str, Any]]:
    scored = []
    for chunk in chunks:
        chunk_symbols, chunk_imports = split_chunk_symbols(chunk, repo_import_names)
        score = weighted_ast_attention_score(query_vars, query_imports, chunk_symbols, chunk_imports, doc_weights)
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]


def run_demo(n_tasks: int = 50, tasks_per_repo: int = 10) -> None:
    tasks = load_tasks(TASKS_PATH)
    print(f"{len(tasks)} tâches chargées depuis {TASKS_PATH.name}")

    seen_per_repo: dict[str, int] = {}
    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    doc_weights_cache: dict[str, dict[str, float]] = {}
    import_names_cache: dict[str, set[str]] = {}

    processed = 0
    for task in tasks:
        if processed >= n_tasks:
            break
        metadata = task["metadata"]
        repo = metadata["task_id"].split("/")[0]
        if seen_per_repo.get(repo, 0) >= tasks_per_repo:
            continue

        repo_dir = REPOS_DIR / repo
        if not repo_dir.exists():
            continue

        if repo not in chunk_cache:
            print(f"\nPréparation de {repo} (découpage AST, IDF, imports)...")
            chunk_cache[repo] = load_and_chunk_repo_ast_cached(str(repo_dir))
            doc_weights_cache[repo] = compute_doc_weights(chunk_cache[repo])
            import_names_cache[repo] = compute_repo_import_names(repo_dir)
            print(f"  -> {len(chunk_cache[repo])} fenêtres, {len(doc_weights_cache[repo])} symboles distincts")

        target_file = repo_dir.joinpath(*metadata["fpath_tuple"][1:])
        try:
            file_source = target_file.read_text(encoding="utf-8")
        except OSError:
            continue

        line_no = metadata["line_no"]
        try:
            module_imports, blocks = build_scope_map(file_source)
        except SyntaxError:
            continue

        # variables candidates : identifiants de tous les blocs qui englobent line_no
        candidate_vars: set[str] = set()
        for block in blocks:
            if block.line_start <= line_no <= block.line_end:
                candidate_vars |= block.identifiers
        candidate_vars -= module_imports  # les imports sont traités séparément

        query_vars = compute_query_vars_with_attention(file_source, line_no, candidate_vars, lam=0.1)
        query_imports = module_imports

        safe_chunks = filter_safe_chunks(chunk_cache[repo], repo_dir, metadata["fpath_tuple"], metadata["context_start_lineno"])
        results = retrieve_top_k_weighted(
            query_vars, query_imports, safe_chunks, import_names_cache[repo], doc_weights_cache[repo], k=3
        )

        seen_per_repo[repo] = seen_per_repo.get(repo, 0) + 1
        processed += 1

        print(f"\n=== {metadata['task_id']} (ligne {line_no}) ===")
        print(f"  query_vars ({len(query_vars)}): {dict(sorted(query_vars.items(), key=lambda x: -x[1])[:5])}")
        print(f"  query_imports ({len(query_imports)}): {sorted(query_imports)[:5]}")
        for rank, result in enumerate(results, start=1):
            rel_path = os.path.relpath(result["file_path"], repo_dir)
            print(f"  #{rank} score={result['score']:.3f}  {rel_path}:{result['line_start']}-{result['line_end']}")


if __name__ == "__main__":
    run_demo(n_tasks=50, tasks_per_repo=10)
