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
        "exact_match": exact_match(completion, record["groundtruth"]),
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
            exact_match(completion, record["groundtruth"]) for completion in completions
        ],
        "edit_similarity": edit_sims,
        "identifier_f1": identifier_f1s,
    }