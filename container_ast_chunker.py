"""Découpage du code par conteneurs AST (fonction/classe), pas par fenêtres
de lignes fixes — via Tree-Sitter, module autonome, non branché à
ast_chunker.py (qui reste la version scope-aware par fenêtres glissantes).

Principe :
- Un noeud fonction/classe/définition décorée assez petit devient UN chunk
  entier (jamais coupé au milieu d'une instruction ou d'une signature).
- Un noeud trop grand est découpé par ses instructions de haut niveau
  internes (if/for/try/... — jamais au milieu d'une instruction), chaque
  sous-chunk conservant l'en-tête (signature + décorateurs) et un léger
  chevauchement avec le sous-chunk précédent pour ne pas perdre les
  variables locales actives déclarées juste au-dessus.
- Le code au niveau module (imports, constantes, `if __name__ == ...`) qui
  n'est ni fonction ni classe est regroupé en chunks "module" séparés, pour
  ne rien perdre silencieusement.
"""

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)

TARGET_NODES = {"function_definition", "class_definition", "decorated_definition"}


def _find_body_node(node):
    """Le noeud 'block' (corps) d'un function_definition/class_definition,
    ou None si le corps est sur la même ligne que la signature (rare)."""
    for child in node.children:
        if child.type == "block":
            return child
    return None


def _header_lines(node, lines):
    """Lignes de la signature (et des décorateurs pour une définition
    décorée), avant le début du corps."""
    if node.type == "decorated_definition":
        inner = node.children[-1]  # le dernier enfant est le function/class_definition
        decorator_end = inner.start_point[0] - 1
        decorator_lines = lines[node.start_point[0]:decorator_end + 1]
        return decorator_lines + _header_lines(inner, lines)

    body = _find_body_node(node)
    if body is None:
        return [lines[node.start_point[0]]]
    header_end = body.start_point[0] - 1
    return lines[node.start_point[0]:header_end + 1]


def _split_large_container(node, lines, max_lines, overlap_lines):
    """Découpe le CORPS d'un noeud trop grand en sous-chunks alignés sur ses
    instructions de haut niveau, avec en-tête conservé + chevauchement."""
    header = _header_lines(node, lines)
    header_text = "\n".join(header)

    body_owner = node.children[-1] if node.type == "decorated_definition" else node
    body = _find_body_node(body_owner)

    if body is None or not body.children:
        # corps sur une seule ligne ou vide -> rien à découper, un seul chunk tel quel
        start_line, end_line = node.start_point[0], node.end_point[0]
        return [{
            "start_line": start_line + 1,
            "end_line": end_line + 1,
            "type": node.type,
            "code": "\n".join(lines[start_line:end_line + 1]),
        }]

    statements = list(body.children)
    budget = max(1, max_lines - len(header))

    groups = []
    current = []
    for stmt in statements:
        if not current:
            current = [stmt]
            continue
        span = stmt.end_point[0] - current[0].start_point[0] + 1
        if span > budget:
            groups.append(current)
            current = [stmt]
        else:
            current.append(stmt)
    if current:
        groups.append(current)

    sub_chunks = []
    prev_end_line = None
    for group in groups:
        group_start = group[0].start_point[0]
        group_end = group[-1].end_point[0]

        overlap_text = []
        if prev_end_line is not None:
            overlap_start = max(0, prev_end_line - overlap_lines + 1)
            # ne pas empiéter sur le début du groupe courant lui-même
            overlap_end = min(prev_end_line, group_start - 1)
            if overlap_end >= overlap_start:
                overlap_text = lines[overlap_start:overlap_end + 1]

        body_text = lines[group_start:group_end + 1]
        code_text = "\n".join([header_text] + overlap_text + body_text)

        sub_chunks.append({
            "start_line": group_start + 1,
            "end_line": group_end + 1,
            "type": f"{node.type}:partial",
            "code": code_text,
        })
        prev_end_line = group_end

    return sub_chunks


def _pack_module_level_lines(nodes, lines, max_lines):
    """Regroupe des instructions consécutives de niveau module (ni fonction
    ni classe) en chunks de type 'module', bornés à max_lines."""
    if not nodes:
        return []
    chunks = []
    current = [nodes[0]]
    for node in nodes[1:]:
        span = node.end_point[0] - current[0].start_point[0] + 1
        if span > max_lines:
            chunks.append(current)
            current = [node]
        else:
            current.append(node)
    chunks.append(current)

    result = []
    for group in chunks:
        start_line, end_line = group[0].start_point[0], group[-1].end_point[0]
        result.append({
            "start_line": start_line + 1,
            "end_line": end_line + 1,
            "type": "module",
            "code": "\n".join(lines[start_line:end_line + 1]),
        })
    return result


def chunk_code_by_ast(code: str, max_lines: int = 30, overlap_lines: int = 3) -> list[dict]:
    """Découpe le code Python en respectant la structure AST (fonctions/classes).

    Ne coupe jamais au milieu d'une instruction ou d'une signature. Un noeud
    trop grand est découpé par ses instructions de haut niveau internes, en
    conservant l'en-tête et un chevauchement de contexte entre sous-chunks.
    Le code de niveau module (hors fonction/classe) est regroupé à part,
    jamais perdu silencieusement.
    """
    if not code.strip():
        return []

    tree = _parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    lines = code.split("\n")
    chunks = []
    orphan_module_nodes = []

    def collect_chunks(node, is_root_level):
        if node.type in TARGET_NODES:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            chunk_len = end_line - start_line + 1

            if chunk_len <= max_lines:
                chunk_text = "\n".join(lines[start_line:end_line + 1])
                chunks.append({
                    "start_line": start_line + 1,
                    "end_line": end_line + 1,
                    "type": node.type,
                    "code": chunk_text,
                })
            else:
                chunks.extend(_split_large_container(node, lines, max_lines, overlap_lines))
            return

        if is_root_level and node is not root_node:
            orphan_module_nodes.append(node)
            return

        for child in node.children:
            collect_chunks(child, is_root_level=(node is root_node))

    collect_chunks(root_node, is_root_level=False)
    chunks.extend(_pack_module_level_lines(orphan_module_nodes, lines, max_lines))
    chunks.sort(key=lambda c: c["start_line"])

    return chunks
