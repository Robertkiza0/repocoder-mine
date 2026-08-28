import ast
import hashlib
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset import SLIDING_STRIDE, SLIDING_WINDOW_SIZE

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "cache" / "ast_chunks"


@dataclass
class ScopeBlock:
    """Un bloc nommé (classe ou fonction/méthode) avec ses bornes de lignes et ses identifiants propres."""

    kind: str  # "class" ou "function"
    name: str
    line_start: int
    line_end: int
    identifiers: set[str] = field(default_factory=set)


class _LocalNamesCollector(ast.NodeVisitor):
    """Collecte les noms assignés/définis dans un scope, sans descendre dans les classes/fonctions imbriquées
    (elles ont leur propre ScopeBlock)."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name)


def _collect_local_names(node: ast.AST) -> set[str]:
    """Noms locaux à un bloc (paramètres, variables assignées, imports locaux, fonctions/classes imbriquées)."""
    collector = _LocalNamesCollector()
    collector.generic_visit(node)  # generic_visit: visite les enfants, pas node lui-même
    return collector.names


def _collect_class_attributes(class_node: ast.ClassDef) -> set[str]:
    """Attributs de classe (`x = 1` dans le corps) et d'instance (`self.x = ...` dans les méthodes)."""
    attributes: set[str] = set()

    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    attributes.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            attributes.add(stmt.target.id)

    for node in ast.walk(class_node):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name):
            attributes.add(node.attr)

    return attributes


def build_scope_map(code: str) -> tuple[set[str], list[ScopeBlock]]:
    """Analyser le code d'un fichier et retourner (imports du module, blocs classes/fonctions).

    Les imports de haut niveau (pas dans une fonction/classe) sont visibles dans
    tout le fichier. Chaque classe et chaque fonction/méthode devient un
    ScopeBlock avec ses propres identifiants (nom, attributs/paramètres,
    variables locales) et ses bornes de lignes (`node.lineno`/`node.end_lineno`).
    """
    tree = ast.parse(code)
    module_imports: set[str] = set()
    blocks: list[ScopeBlock] = []

    def visit(node: ast.AST, inside_def: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)) and not inside_def:
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        module_imports.add(alias.asname or alias.name.split(".")[0])
                else:
                    for alias in child.names:
                        module_imports.add(alias.asname or alias.name)

            if isinstance(child, ast.ClassDef):
                own_methods = {
                    n.name for n in child.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                identifiers = {child.name} | _collect_class_attributes(child) | own_methods
                blocks.append(ScopeBlock("class", child.name, child.lineno, child.end_lineno, identifiers))
                visit(child, True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                identifiers = {child.name} | _collect_local_names(child)
                blocks.append(ScopeBlock("function", child.name, child.lineno, child.end_lineno, identifiers))
                visit(child, True)
            else:
                visit(child, inside_def)

    visit(tree, False)
    return module_imports, blocks


def chunk_file_ast(
    file_path: str,
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
) -> list[dict[str, Any]]:
    """Découper un fichier Python en fenêtres glissantes de lignes enrichies par l'AST.

    Chaque fenêtre hérite des identifiants de TOUS les blocs (classe et/ou
    fonction(s)) dont l'intervalle de lignes chevauche la fenêtre, plus les
    imports du module. Une fenêtre à cheval sur deux méthodes hérite ainsi
    des identifiants des deux, afin qu'une requête touchant l'une ou l'autre
    puisse retrouver ce morceau.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            code = file.read()
    except OSError as error:
        print(f"Erreur de lecture de {file_path}: {error}")
        return []

    try:
        module_imports, blocks = build_scope_map(code)
    except SyntaxError as error:
        print(f"Fichier ignoré (syntaxe invalide) {file_path}: {error}")
        return []

    lines = code.splitlines()
    if not lines:
        return []

    chunks = []
    for start in range(0, len(lines), stride):
        end = min(start + window_size, len(lines))
        raw_code = "\n".join(lines[start:end]).strip()

        if raw_code:
            line_start, line_end = start + 1, end  # lignes 1-indexées, comme node.lineno

            identifiers = set(module_imports)
            for block in blocks:
                if max(line_start, block.line_start) <= min(line_end, block.line_end):
                    identifiers |= block.identifiers

            chunks.append(
                {
                    "file_path": file_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "raw_code": raw_code,
                    "identifiers": sorted(identifiers),
                }
            )

        if end >= len(lines):
            break

    return chunks


def load_and_chunk_repo_ast(
    dir_path: str,
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
) -> list[dict[str, Any]]:
    """Parcourir tous les fichiers .py d'un dossier et produire les chunks enrichis par AST."""
    all_chunks: list[dict[str, Any]] = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                all_chunks.extend(chunk_file_ast(file_path, window_size=window_size, stride=stride))
    return all_chunks


def _repo_fingerprint(dir_path: str) -> str:
    """Empreinte du contenu d'un dossier (chemin + date de modif + taille de chaque .py).

    Sert à invalider le cache automatiquement si un fichier source a changé,
    sans avoir à relire/hacher le contenu de chaque fichier.
    """
    entries = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                stat = os.stat(file_path)
                entries.append((os.path.relpath(file_path, dir_path), stat.st_mtime_ns, stat.st_size))
    entries.sort()
    return hashlib.sha1(repr(entries).encode("utf-8")).hexdigest()


def _cache_path(dir_path: str, window_size: int, stride: int, cache_dir: str | Path) -> Path:
    safe_name = re.sub(r"[^\w.-]", "_", os.path.normpath(os.path.abspath(dir_path)))
    return Path(cache_dir) / f"{safe_name}_ws{window_size}_stride{stride}.pkl"


def load_and_chunk_repo_ast_cached(
    dir_path: str,
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    """Comme `load_and_chunk_repo_ast`, mais met le résultat en cache sur disque.

    Le cache est invalidé automatiquement si un fichier .py du dossier a été
    ajouté/modifié/supprimé depuis la dernière exécution (voir `_repo_fingerprint`),
    ou si `window_size`/`stride` changent (chaque combinaison a son propre fichier
    de cache).
    """
    cache_file = _cache_path(dir_path, window_size, stride, cache_dir)
    fingerprint = _repo_fingerprint(dir_path)

    if cache_file.exists():
        with cache_file.open("rb") as file:
            cached = pickle.load(file)
        if cached.get("fingerprint") == fingerprint:
            return cached["chunks"]

    chunks = load_and_chunk_repo_ast(dir_path, window_size=window_size, stride=stride)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as file:
        pickle.dump({"fingerprint": fingerprint, "chunks": chunks}, file)
    return chunks


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    demo_file = os.path.join(project_dir, "retriever.py")
    chunks = chunk_file_ast(demo_file)

    print(f"{len(chunks)} morceaux générés depuis {demo_file}\n")
    for chunk in chunks[:5]:
        print(f"--- lignes {chunk['line_start']}-{chunk['line_end']} ---")
        print("identifiants:", chunk["identifiers"])
        print()
