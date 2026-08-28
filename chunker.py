import os

from dataset import load_cceval_dataset


def chunk_file(file_path: str, chunk_size: int = 20, overlap: int = 10) -> list[str]:
    """Lire un fichier et le découper en morceaux de lignes qui se chevauchent."""
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être supérieur à 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap doit être compris entre 0 et chunk_size - 1")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except OSError as error:
        print(f"Erreur de lecture de {file_path}: {error}")
        return []

    if not lines:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(lines), step):
        chunk_text = "".join(lines[start:start + chunk_size]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if start + chunk_size >= len(lines):
            break
    return chunks


def chunk_dataset(
    records: list[dict], chunk_size: int = 20, overlap: int = 10
) -> list[dict]:
    """Découper les prompts en gardant les réponses attendues."""
    chunks = []
    step = chunk_size - overlap

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size et overlap ne sont pas valides")

    for record in records:
        prompt_lines = record["prompt"].splitlines(keepends=True)
        for start in range(0, len(prompt_lines), step):
            prompt_chunk = "".join(prompt_lines[start:start + chunk_size]).strip()
            if prompt_chunk:
                chunks.append(
                    {
                        "task_id": record.get("metadata", {}).get("task_id"),
                        "prompt_chunk": prompt_chunk,
                        "groundtruth": record["groundtruth"],
                        "right_context": record["right_context"],
                        "metadata": record.get("metadata", {}),
                    }
                )
            if start + chunk_size >= len(prompt_lines):
                break

    return chunks


def load_and_chunk_repo(
    dir_path: str, chunk_size: int = 20, overlap: int = 10
) -> dict[str, list[str]]:
    """Lire tous les fichiers Python d'un dossier et les découper."""
    all_chunks = {}
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                all_chunks[file_path] = chunk_file(
                    file_path, chunk_size=chunk_size, overlap=overlap
                )
    return all_chunks


def load_and_chunk_jsonl(
    file_path: str, chunk_size: int = 20, overlap: int = 10
) -> list[dict]:
    """Charger le dataset puis découper ses prompts."""
    records = load_cceval_dataset(path=file_path)
    return chunk_dataset(records, chunk_size=chunk_size, overlap=overlap)


if __name__ == "__main__":
    jsonl_path = r"C:\Users\User\Desktop\cceval\data\python\line_completion.jsonl"
    chunks = load_and_chunk_jsonl(jsonl_path, chunk_size=5, overlap=2)

    print(f"Morceaux créés depuis le fichier JSONL : {len(chunks)}")
    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Morceau {index} ---")
        print("Task ID :", chunk["task_id"])
        print("Prompt :\n", chunk["prompt_chunk"])
        print("Groundtruth :", chunk["groundtruth"])
