from typing import Any

from thefuzz import fuzz

from cceval.base import Example
from cceval.metrics import MetricCalculator


def exact_match(completion: str, groundtruth: str) -> bool:
    """Vérifier si la complétion correspond exactement au groundtruth (espaces en bord ignorés)."""
    return completion.strip() == groundtruth.strip()


def edit_similarity(completion: str, groundtruth: str) -> float:
    """Similarité d'édition (Levenshtein, 0 à 100) entre la complétion et le groundtruth."""
    return fuzz.ratio(completion.strip(), groundtruth.strip())


def _postprocessed_exact_match(
    calculator: MetricCalculator, prompt: str, completion: str, groundtruth: str
) -> bool:
    """exact_match cohérent avec edit_similarity/identifier_f1.

    `calculator.compute_metrics` compare la complétion après post-traitement
    (tronquée à la première instruction complète, commentaires retirés) — pas
    le texte brut multi-lignes généré par le modèle. Comparer `exact_match`
    sur le texte brut donnait des faux négatifs (edit_similarity=100 mais
    exact_match=False) : on applique donc le même post-traitement ici.
    """
    processed_completion = calculator._postprocess(prompt, completion)
    processed_groundtruth = calculator._remove_comments(groundtruth)
    return processed_completion.strip() == processed_groundtruth.strip()


def evaluate_completion(
    record: dict[str, Any], completion: str, language: str = "python"
) -> dict[str, Any]:
    """Comparer un code généré avec le groundtruth CrossCodeEval."""
    example = Example.model_validate(record)
    calculator = MetricCalculator(lang=language)
    edit_sims, identifier_f1s = calculator.compute_metrics(
        example, [completion]
    )
    return {
        "exact_match": _postprocessed_exact_match(
            calculator, example.prompt, completion, record["groundtruth"]
        ),
        "edit_similarity": edit_sims[0],
        "identifier_f1": identifier_f1s[0],
    }


def evaluate_completions(
    record: dict[str, Any],
    completions: list[str],
    language: str = "python",
) -> dict[str, list[Any]]:
    """Évaluer plusieurs propositions pour un même exemple."""
    example = Example.model_validate(record)
    calculator = MetricCalculator(lang=language)
    edit_sims, identifier_f1s = calculator.compute_metrics(example, completions)
    return {
        "exact_match": [
            _postprocessed_exact_match(calculator, example.prompt, completion, record["groundtruth"])
            for completion in completions
        ],
        "edit_similarity": edit_sims,
        "identifier_f1": identifier_f1s,
    }
