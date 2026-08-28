import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def best_snippet(request, snippets):
    vectorizer = TfidfVectorizer()
    matrice_repo_snippets = vectorizer.fit_transform(snippets)
    matrice_request = vectorizer.transform([request])
    scores = cosine_similarity(matrice_request, matrice_repo_snippets)[0]
    best_index = scores.argmax()
    return snippets[best_index], scores[best_index]


def tokenize_code(code: str) -> set[str]:
    """Extraire le sac de mots (bag of words) d'un extrait de code."""
    return set(re.findall(r"\w+", code))


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Calculer l'indice de Jaccard entre deux sacs de tokens."""
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def retrieve_top_k_jaccard(
    incomplete_code: str, snippets: list[dict[str, str]], k: int = 10
) -> list[dict[str, Any]]:
    """Classer les blocs de code par indice de Jaccard avec le code incomplet.

    Compare le sac de tokens du code incomplet à celui de chaque bloc
    (`{"file", "snippet"}`) et retourne les k meilleurs
    `{"file", "snippet", "score"}`, triés par score décroissant.
    """
    query_tokens = tokenize_code(incomplete_code)

    scored_snippets = [
        {
            "file": snippet.get("file"),
            "snippet": snippet["snippet"],
            "score": jaccard_similarity(query_tokens, tokenize_code(snippet["snippet"])),
        }
        for snippet in snippets
    ]
    scored_snippets.sort(key=lambda item: item["score"], reverse=True)
    return scored_snippets[:k]


def load_repository_snippets(
    repository: str, repositories_dir: str | Path
) -> list[dict[str, str]]:
    """Charger les blocs de code sauvegardés (dataset.save_repository_snippets) d'un dépôt."""
    safe_name = re.sub(r"[^\w.-]", "_", repository)
    repository_path = Path(repositories_dir) / f"{safe_name}.jsonl"

    snippets = []
    with repository_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                snippets.append({"file": record.get("file"), "snippet": record["snippet"]})
    return snippets


def retrieve_top_k_from_repository(
    incomplete_code: str,
    repository: str,
    repositories_dir: str | Path,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Charger les blocs de code d'un dépôt puis retourner les k meilleurs (Jaccard)."""
    snippets = load_repository_snippets(repository, repositories_dir)
    return retrieve_top_k_jaccard(incomplete_code, snippets, k=k)


def retrieve_top_k_for_dataset(
    records: list[dict[str, Any]],
    repositories_dir: str | Path,
    k: int = 10,
    query_lines: int | None = None,
) -> list[dict[str, Any]]:
    """Appliquer le retriever Jaccard à chaque exemple de line_completion.jsonl.

    La requête vient de `record["prompt"]` (le code incomplet), jamais du
    groundtruth ; si `query_lines` est fourni, seules les `query_lines`
    dernières lignes du prompt sont utilisées (cf. `dataset.last_lines`, S_s
    dans l'article RepoCoder). Les blocs de chaque dépôt sont tokenisés une
    seule fois (mis en cache) pour éviter de retokeniser à chaque exemple.
    """
    from dataset import last_lines

    results = []
    repository_index: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        metadata = record.get("metadata", {})
        repository = metadata.get("repository")
        if not repository:
            continue

        if repository not in repository_index:
            snippets = load_repository_snippets(repository, repositories_dir)
            repository_index[repository] = [
                {"file": snippet["file"], "snippet": snippet["snippet"], "tokens": tokenize_code(snippet["snippet"])}
                for snippet in snippets
            ]

        query = record["prompt"] if query_lines is None else last_lines(record["prompt"], query_lines)
        query_tokens = tokenize_code(query)
        scored_snippets = [
            {
                "file": snippet["file"],
                "snippet": snippet["snippet"],
                "score": jaccard_similarity(query_tokens, snippet["tokens"]),
            }
            for snippet in repository_index[repository]
        ]
        scored_snippets.sort(key=lambda item: item["score"], reverse=True)

        results.append(
            {
                "task_id": metadata.get("task_id"),
                "repository": repository,
                "retrieved_chunks": scored_snippets[:k],
            }
        )

    return results


if __name__ == "__main__":
    from dataset import SLIDING_STRIDE, load_jsonl, save_jsonl

    project_dir = Path(__file__).resolve().parent
    records = load_jsonl(r"C:\Users\User\Downloads\line_completion.jsonl")
    results = retrieve_top_k_for_dataset(
        records,
        project_dir / "data" / "repositories",
        k=10,
        query_lines=SLIDING_STRIDE,
    )
    save_jsonl(results, project_dir / "data" / "retrieved_jaccard.jsonl")

    print(f"{len(results)} exemples traités et sauvegardés dans data/retrieved_jaccard.jsonl")