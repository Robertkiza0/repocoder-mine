from typing import Any

from retriever import best_snippet


def prompt_builder(request: str, snippets: list[str]) -> str:
    """Construire un prompt avec l'extrait le plus similaire."""
    if not snippets:
        return f"# Demande de l'utilisateur\n{request}"

    snippet, score = best_snippet(request, snippets)
    return (
        f"# Extrait similaire (score: {score:.3f})\n"
        f"{snippet}\n\n"
        f"# Demande de l'utilisateur\n{request}"
    )


def prompt_from_record(record: dict[str, Any]) -> str:
    """Construire un prompt à partir d'un exemple du dataset."""
    return prompt_builder(record["prompt"], [record["prompt"]])


def format_retrieved_chunks(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Formater les extraits récupérés (Jaccard) en un bloc de contexte pour l'IA.

    Les extraits sont placés par score croissant : le moins pertinent en
    premier, le plus pertinent en dernier (donc le plus proche du code à
    compléter). Chaque extrait est précédé du chemin de son fichier.
    """
    ordered_chunks = sorted(retrieved_chunks, key=lambda chunk: chunk["score"])

    blocks = [
        f"# Extrait trouvé dans : {chunk['file']} (score: {chunk['score']:.3f})\n{chunk['snippet']}"
        for chunk in ordered_chunks
    ]
    return "\n\n".join(blocks)


def build_prompt_from_retrieval(incomplete_code: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """Construire le prompt final pour l'IA : contexte récupéré (croissant) + code incomplet."""
    context = format_retrieved_chunks(retrieved_chunks)
    if not context:
        return incomplete_code

    return f"{context}\n\n{incomplete_code}"
