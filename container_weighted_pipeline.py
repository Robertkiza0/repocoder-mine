"""Adapte container_ast_chunker (découpage par conteneurs AST : fonctions/
classes entières ou découpées par bloc interne) au format attendu par
weighted_ast_attention_score et par les fonctions communes du pipeline
(filter_safe_chunks, build_prompt, ...) : file_path, line_start, line_end,
raw_code, identifiers, chunk_imports.

Contrairement à ast_chunker (fenêtres de lignes fixes), chaque chunk produit
ici est un extrait de code généralement syntaxiquement valide seul (fonction/
méthode entière, ou en-tête + suite de statements complets) — on peut donc
réellement parser chaque chunk avec ast.parse pour en extraire ses propres
symboles/imports, sans avoir besoin de l'approximation "ensemble des imports
du dépôt entier" utilisée pour les chunks ast_chunker.
"""

import ast
import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Any

from container_ast_chunker import chunk_code_by_ast

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "cache" / "container_chunks"

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_]\w*")


def extract_chunk_identifiers(code_text: str) -> tuple[set[str], set[str]]:
    """(symboles, imports) d'un chunk. Tente un vrai parse AST (les chunks de
    container_ast_chunker sont généralement du Python valide isolément) ;
    retombe sur une extraction par regex si le parse échoue (ex. légère
    incohérence de recombinaison en-tête + chevauchement)."""
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        tokens = set(_IDENTIFIER_PATTERN.findall(code_text))
        return tokens, set()

    imports: set[str] = set()
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.arg):
            symbols.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)

    symbols -= imports
    return symbols, imports


def chunk_file_container_ast(
    file_path: str, max_lines: int = 30, overlap_lines: int = 3
) -> list[dict[str, Any]]:
    try:
        code = open(file_path, "r", encoding="utf-8").read()
    except OSError as error:
        print(f"Erreur de lecture de {file_path}: {error}")
        return []

    try:
        raw_chunks = chunk_code_by_ast(code, max_lines=max_lines, overlap_lines=overlap_lines)
    except SyntaxError as error:
        print(f"Fichier ignoré (syntaxe invalide) {file_path}: {error}")
        return []

    chunks = []
    for raw_chunk in raw_chunks:
        symbols, imports = extract_chunk_identifiers(raw_chunk["code"])
        chunks.append({
            "file_path": file_path,
            "line_start": raw_chunk["start_line"],
            "line_end": raw_chunk["end_line"],
            "raw_code": raw_chunk["code"],
            "identifiers": sorted(symbols | imports),  # pour compute_doc_weights (IDF), comme ast_chunker
            "chunk_imports": sorted(imports),
        })
    return chunks


def load_and_chunk_repo_container_ast(
    dir_path: str, max_lines: int = 30, overlap_lines: int = 3
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                all_chunks.extend(chunk_file_container_ast(file_path, max_lines=max_lines, overlap_lines=overlap_lines))
    return all_chunks


def _repo_fingerprint(dir_path: str) -> str:
    entries = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                stat = os.stat(file_path)
                entries.append((os.path.relpath(file_path, dir_path), stat.st_mtime_ns, stat.st_size))
    entries.sort()
    return hashlib.sha1(repr(entries).encode("utf-8")).hexdigest()


def load_and_chunk_repo_container_ast_cached(
    dir_path: str,
    max_lines: int = 30,
    overlap_lines: int = 3,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    safe_name = re.sub(r"[^\w.-]", "_", os.path.normpath(os.path.abspath(dir_path)))
    cache_file = Path(cache_dir) / f"{safe_name}_max{max_lines}_ov{overlap_lines}.pkl"
    fingerprint = _repo_fingerprint(dir_path)

    if cache_file.exists():
        with cache_file.open("rb") as file:
            cached = pickle.load(file)
        if cached.get("fingerprint") == fingerprint:
            return cached["chunks"]

    chunks = load_and_chunk_repo_container_ast(dir_path, max_lines=max_lines, overlap_lines=overlap_lines)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as file:
        pickle.dump({"fingerprint": fingerprint, "chunks": chunks}, file)
    return chunks


def retrieve_top_k_weighted_container(
    query_vars: dict[str, float],
    query_imports: set[str],
    chunks: list[dict[str, Any]],
    doc_weights: dict[str, float],
    k: int = 10,
    var_weight: float = 2.0,
    import_weight: float = 2.5,
) -> list[dict[str, Any]]:
    """Comme retrieve_top_k_weighted, mais les imports du chunk viennent de
    son propre parse (chunk['chunk_imports']), pas d'une approximation par
    dépôt entier."""
    from weighted_ast_scorer import weighted_ast_attention_score

    scored = []
    for chunk in chunks:
        chunk_symbols = set(chunk["identifiers"]) - set(chunk["chunk_imports"])
        chunk_imports = set(chunk["chunk_imports"])
        score = weighted_ast_attention_score(
            query_vars, query_imports, chunk_symbols, chunk_imports, doc_weights,
            var_weight=var_weight, import_weight=import_weight,
        )
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]
