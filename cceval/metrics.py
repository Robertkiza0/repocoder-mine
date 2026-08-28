"""CrossCodeEval completion 평가 지표 (edit similarity, identifier F1)."""

from __future__ import annotations

import re
from typing import Iterator

import tree_sitter_c_sharp as tscsharp
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from thefuzz import fuzz
from tree_sitter import Language, Node, Parser

from .base import Example


class MetricCalculator:
    def __init__(self, lang: str = "python"):
        self.lang = lang

        if lang == "python":
            self.parser = Parser(Language(tspython.language()))
        elif lang == "java":
            self.parser = Parser(Language(tsjava.language()))
        elif lang == "csharp":
            self.parser = Parser(Language(tscsharp.language()))
        elif lang == "typescript":
            self.parser = Parser(Language(tstypescript.language_typescript()))
        else:
            raise ValueError

    def compute_metrics(
        self, example: Example, completions: list[str]
    ) -> tuple[list[float], list[float]]:
        """completion 리스트에 대해 edit_similarity, identifier_f1을 계산한다."""
        target = self._remove_comments(example.groundtruth)
        target_ids = self.extract_identifiers(target)

        edit_sims, id_f1s = [], []
        for comp in completions:
            # postprocess completion (첫 statement만 추출)
            comp = self._postprocess(example.prompt, comp)

            # compute edit similarity
            comp_clean = self._remove_comments(comp)
            edit_sim = fuzz.ratio(comp_clean, target)
            edit_sims.append(edit_sim)

            # compute identifier f1
            comp_ids = self.extract_identifiers(comp_clean)
            id_f1 = compute_multilabel_metrics(target_ids, comp_ids)["f1"] * 100
            id_f1s.append(id_f1)
        return edit_sims, id_f1s

    def _postprocess(self, prompt: str, completion: str) -> str:
        """생성된 코드를 후처리한다 (주석 제거, 공백 정리)."""
        completion = self._truncate_code_lines(prompt, completion)
        completion = self._remove_comments(completion)
        return completion

    def _truncate_code_lines(self, prompt: str, completion: str) -> str:
        """
        주어진 completion에서 완성된 line 까지만 추출하는 함수

        Args:
            prompt (str): 주어진 prompt
            completion (str): 주어진 completion

        Returns:
        str: 완성된 line 까지만 추출된 completion
        """
        if self.lang == "python":
            # python은 문장이 완성된 line
            return _cut_first_statement_completion(prompt, completion, self.parser)
        elif self.lang in ["java", "csharp", "typescript"]:
            # java, csharp, typescript는 {}로 감싸진 문장이 완성된 line
            return _get_bracket_lang_statement(completion)
        else:
            raise ValueError

    def _remove_comments(self, code: str) -> str:
        """코드에서 주석을 제거한다."""
        if self.lang == "python":
            return re.sub(r"#.*", "", code)
        # Java, C#, TypeScript
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        return code

    def extract_identifiers(self, code: str) -> set[str]:
        """코드에서 identifier 텍스트 집합을 추출한다."""
        code_bytes = code.encode("utf8")
        tree = self.parser.parse(code_bytes)
        return get_identifier_texts(tree.root_node, code_bytes)


def get_identifier_texts(node: Node, code_bytes: bytes) -> set[str]:
    """이 node에 포함된 하위 identifier 노드들에 대해서, identifier 노드들의 text를 반환합니다. identifier의 텍스트 집합에서 중복을 제거한 뒤 반환합니다."""
    identifier_nodes = get_nodes_by_type(node, "identifier")
    identifier_texts = [
        get_text_from_node(node, code_bytes) for node in identifier_nodes
    ]
    identifier_texts = set(identifier_texts)
    return identifier_texts


def get_nodes_by_type(node: Node, type_name: str) -> list[Node]:
    return [n for n in walk_tree(node) if n.type == type_name]


def walk_tree(node: Node) -> Iterator[Node]:
    """tree-sitter 노드를 pre-order로 순회하는 제너레이터."""
    yield node
    for child in node.children:
        yield from walk_tree(child)


def get_text_from_node(node: Node, code_bytes: bytes) -> str:
    return code_bytes[node.start_byte : node.end_byte].decode("utf8")


# 이름으로 사용되는 노드 타입
NAME_NODE_TYPES = {"identifier", "type_identifier", "property_identifier"}


def compute_multilabel_metrics(
    gt: set[str],
    pred: set[str],
) -> dict[str, float]:
    """multi-label precision, recall, F1을 계산한다.

    Args:
        gt: ground truth 라벨 집합
        pred: predicted 라벨 집합

    Returns:
        {"precision": ..., "recall": ..., "f1": ...}
    """
    if not gt and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(gt & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def _get_bracket_lang_statement(completion):
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[: end_idx + 1] if end_idx else completion


def _get_index(lines: list[str], point: tuple):
    index = sum(len(lines[i]) for i in range(point[0])) + point[1]
    return index


def _cut_first_statement_completion(
    prompt: str, completion: str, parser: Parser
) -> str:
    """
    (prompt + completion)에서 'prompt 마지막 지점부터 시작하는 첫 statement'가 끝나는 지점까지를 찾고,
    그 구간에 해당하는 completion 부분만 잘라 반환합니다.

    completion이 불완전해서 첫 statement 종료를 못 찾으면 completion 원문을 반환합니다.
    """
    if len(completion.strip()) == 0:
        return completion

    # prompt의 마지막 10라인 정도만 사용해서 파싱
    line_idx = 10
    prompt = "".join(prompt.splitlines(keepends=True)[-line_idx:])

    # prompt에 여러 줄 주석이 있을 경우, 그 주석이 끝나는 지점까지 덜 포함
    while '"""' in prompt or "'''" in prompt:
        line_idx -= 1
        prompt = "".join(prompt.splitlines(keepends=True)[-line_idx:])

    text = prompt + completion
    lines = text.splitlines(keepends=True)
    tree = parser.parse(text.encode("utf8"))
    node = tree.root_node
    while node:
        assert _get_index(lines, node.start_point) <= len(
            prompt
        ), "No valid first node found"
        assert _get_index(lines, node.end_point) > len(
            prompt
        ), "No valid first node found"
        if node.type.endswith("_statement"):
            if node.type not in [
                "for_statement",
                "if_statement",
                "with_statement",
                "while_statement",
                "try_statement",
                "match_statement",
            ]:
                break
        if node.type == "argument_list":
            break
        if node.type == "parameters":
            break
        if node.type == "case_pattern":
            break
        if node.type == "attribute":
            break
        if node.type == "comparison_operator":
            break
        if len(node.children) == 0:
            break
        next_node = None
        for child in node.children:
            start_index = _get_index(lines, child.start_point)
            end_index = _get_index(lines, child.end_point)
            if start_index <= len(prompt) and end_index > len(prompt):
                next_node = child
                break
        if next_node is None:
            break
        node = next_node

    return text[len(prompt) : _get_index(lines, node.end_point)]
