"""Distance de sauts dans l'arbre AST, pour calculer l'attention statique
(attention_score = exp(-lambda * distance_ast)) attendue par
weighted_ast_scorer.weighted_ast_attention_score.

Module autonome : parse le VRAI fichier source complet (pas un extrait
tronqué, souvent syntaxiquement invalide isolément) et mesure la distance
entre l'occurrence la plus proche (avant le curseur) d'une variable et le
noeud englobant le curseur, via leur plus proche ancêtre commun (LCA).
"""

import ast
import math


def annotate_tree(root: ast.AST) -> tuple[dict[int, ast.AST], dict[int, int]]:
    """Parcourt l'arbre une fois et retourne (parent, depth), indexés par id(node).

    id(node) plutôt que node directement : les noeuds ast ne sont pas hashables
    de façon fiable pour tous les types, id() est stable et rapide.
    """
    parent: dict[int, ast.AST] = {}
    depth: dict[int, int] = {}
    stack: list[tuple[ast.AST, ast.AST | None, int]] = [(root, None, 0)]
    while stack:
        node, par, d = stack.pop()
        parent[id(node)] = par
        depth[id(node)] = d
        for child in ast.iter_child_nodes(node):
            stack.append((child, node, d + 1))
    return parent, depth


def find_cursor_node(root: ast.AST, depth: dict[int, int], line_no: int) -> ast.AST:
    """Le noeud le plus profond (le plus spécifique) dont l'intervalle de lignes
    contient line_no — représente \"où se trouve le curseur\" dans l'arbre."""
    best = root
    best_depth = -1
    for node in ast.walk(root):
        node_start = getattr(node, "lineno", None)
        node_end = getattr(node, "end_lineno", None)
        if node_start is not None and node_end is not None and node_start <= line_no <= node_end:
            d = depth[id(node)]
            if d > best_depth:
                best = node
                best_depth = d
    return best


def find_closest_occurrence(root: ast.AST, var_name: str, line_no: int) -> ast.AST | None:
    """La dernière occurrence (Name ou arg) de var_name strictement avant line_no.

    \"Dernière avant le curseur\" = la définition/usage le plus probablement
    pertinent pour compléter le code à cet endroit.
    """
    best_node = None
    best_line = -1
    for node in ast.walk(root):
        name = None
        if isinstance(node, ast.Name) and node.id == var_name:
            name = node.id
        elif isinstance(node, ast.arg) and node.arg == var_name:
            name = node.arg
        if name is None:
            continue
        node_line = getattr(node, "lineno", None)
        if node_line is not None and node_line <= line_no and node_line > best_line:
            best_line = node_line
            best_node = node
    return best_node


def tree_distance(
    node_a: ast.AST, node_b: ast.AST, parent: dict[int, ast.AST], depth: dict[int, int]
) -> int:
    """Nombre de sauts entre deux noeuds dans l'arbre, via leur plus proche
    ancêtre commun (LCA) : depth(a) + depth(b) - 2*depth(LCA)."""
    ancestors_a = set()
    node = node_a
    while node is not None:
        ancestors_a.add(id(node))
        node = parent.get(id(node))

    node = node_b
    steps_b = 0
    while node is not None and id(node) not in ancestors_a:
        node = parent.get(id(node))
        steps_b += 1

    if node is None:
        # pas d'ancêtre commun trouvé (ne devrait pas arriver dans un même arbre
        # connexe) -> distance maximale plutôt que planter
        return depth[id(node_a)] + depth[id(node_b)]

    lca_depth = depth[id(node)]
    return (depth[id(node_a)] - lca_depth) + steps_b


def compute_query_vars_with_attention(
    file_source: str,
    line_no: int,
    candidate_vars: set[str],
    lam: float = 0.1,
) -> dict[str, float]:
    """Pour chaque variable candidate présente avant line_no dans le vrai fichier,
    calcule attention_score = exp(-lambda * distance_ast) où distance_ast est la
    distance de sauts (LCA) entre sa dernière occurrence et le noeud du curseur.

    Une variable candidate absente du fichier (ou jamais utilisée avant line_no)
    est simplement omise du résultat (pas de score par défaut).
    """
    root = ast.parse(file_source)
    parent, depth = annotate_tree(root)
    cursor_node = find_cursor_node(root, depth, line_no)

    query_vars: dict[str, float] = {}
    for var_name in candidate_vars:
        occurrence = find_closest_occurrence(root, var_name, line_no)
        if occurrence is None:
            continue
        distance = tree_distance(occurrence, cursor_node, parent, depth)
        query_vars[var_name] = math.exp(-lam * distance)

    return query_vars
