from pathlib import Path
from typing import Any, Callable

from dataset import SLIDING_STRIDE, SLIDING_WINDOW_SIZE, first_lines, last_lines
from prompt_builder import build_prompt_from_retrieval
from retriever import retrieve_top_k_from_repository


def run_repocoder_pipeline(
    record: dict[str, Any],
    repositories_dir: str | Path,
    generate: Callable[[str], str],
    k: int = 10,
    iterations: int = 2,
    window_size: int = SLIDING_WINDOW_SIZE,
    stride: int = SLIDING_STRIDE,
) -> str:
    """Pipeline RepoCoder itératif (retrieval + génération) pour un exemple CCEval.

    `generate` est la fonction de génération à utiliser (ex.
    `generator.call_model_api`) — injectée en paramètre pour que ce module
    reste indépendant de l'orchestrateur.

    Itération 1 : la requête est constituée des `window_size` dernières
    lignes du prompt (code incomplet), jamais du groundtruth.

    Itérations suivantes : la requête est reconstruite en concaténant les
    (window_size - stride) dernières lignes du prompt avec les `stride`
    premières lignes de la prédiction générée à l'itération précédente —
    exactement comme dans RepoCoder, où la prédiction précédente sert
    d'indice pour retrouver un contexte plus pertinent.

    À chaque itération : retrieval Jaccard (k meilleurs), prompt formaté
    (extraits en ordre croissant + chemin de fichier), puis génération. La
    dernière prédiction est retournée.
    """
    metadata = record.get("metadata", {})
    repository = metadata.get("repository")
    task_id = metadata.get("task_id")
    incomplete_code = record["prompt"]

    query = last_lines(incomplete_code, window_size)
    completion = ""

    for _ in range(iterations):
        retrieved_chunks = retrieve_top_k_from_repository(
            query, repository, repositories_dir, k=k, task_id=task_id
        )
        prompt = build_prompt_from_retrieval(incomplete_code, retrieved_chunks)
        completion = generate(prompt)

        query = last_lines(incomplete_code, window_size - stride) + first_lines(completion, stride)

    return completion
