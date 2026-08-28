import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour que `cceval` soit importable (utilisé par metrics.py)

import requests
from dotenv import load_dotenv
from groq import Groq

from dataset import load_jsonl, save_jsonl
from iterate import run_repocoder_pipeline
from metrics import evaluate_completion

load_dotenv()  # charge GROQ_API_KEY depuis le .env à la racine du repo

SYSTEM_PROMPT = (
    "You are a code completion engine. Continue the code exactly where it "
    "stops. Reply with only the missing code, no explanation, no markdown."
)


def call_model_api(
    prompt: str,
    model: str = "openai/gpt-oss-20b",
    max_completion_tokens: int = 256,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    api_key: str | None = None,
) -> str:
    """Appeler l'API Groq pour compléter le prompt.

    La clé API est lue dans la variable d'environnement GROQ_API_KEY si
    `api_key` n'est pas fourni. `gpt-oss-20b` est un modèle "reasoning" : son
    raisonnement interne consomme le budget de tokens avant la réponse, d'où
    `reasoning_effort="low"` et une marge de tokens plus large par défaut.
    """
    client = Groq(api_key=api_key) if api_key else Groq()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content or ""


def call_ollama_api(
    prompt: str,
    model: str = "starcoder2:3b",
    base_url: str = "http://localhost:11434",
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: float = 600,
) -> str:
    """Appeler un modèle Ollama local pour compléter le prompt.

    Utilise `/api/generate` (complétion brute), pas `/api/chat` : les
    modèles de code comme starcoder2 ne sont pas instruction-tunés (capacités
    "completion"/"insert", pas "chat") — un format chat dégraderait la sortie.
    En CPU, un prompt de ~3000 tokens (contexte récupéré inclus) peut prendre
    plusieurs minutes, d'où le timeout large.
    """
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


_hf_model_cache: dict[str, tuple[Any, Any]] = {}


def call_huggingface_api(
    prompt: str,
    model: str = "bigcode/starcoderbase-7b",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    max_prompt_tokens: int = 4096,
) -> str:
    """Générer une complétion avec un modèle Hugging Face chargé localement.

    Le modèle et le tokenizer sont chargés une seule fois par nom de modèle
    (mis en cache) — utile sur Colab où `run_experiment` appelle cette
    fonction pour chaque itération de chaque exemple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model not in _hf_model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        hf_model.eval()
        _hf_model_cache[model] = (tokenizer, hf_model)

    tokenizer, hf_model = _hf_model_cache[model]
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens
    ).to(hf_model.device)

    with torch.no_grad():
        output_ids = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generer_code(prompt: str) -> str:
    """Point d'entrée du générateur de code : appelle l'API du modèle configuré."""
    return call_model_api(prompt)


def run_experiment(
    line_completion_path: str | Path,
    repositories_dir: str | Path,
    output_path: str | Path,
    k: int = 10,
    iterations: int = 2,
    sample_size: int | None = 20,
    generate: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Orchestrer l'expérimentation complète : dataset -> retrieval -> génération -> évaluation.

    Charge `line_completion.jsonl`, lance le pipeline RepoCoder itératif
    (`iterate.run_repocoder_pipeline`) sur chaque exemple avec `generate`
    (par défaut `call_model_api`) comme générateur, évalue chaque résultat
    (`metrics.evaluate_completion`) et sauvegarde tout en JSONL.
    """
    generate = generate or call_model_api
    records = load_jsonl(line_completion_path)
    if sample_size is not None:
        records = records[:sample_size]

    results = []
    try:
        for record in records:
            completion = run_repocoder_pipeline(
                record, repositories_dir, generate=generate, k=k, iterations=iterations
            )
            scores = evaluate_completion(record, completion)
            results.append(
                {
                    "task_id": record["metadata"]["task_id"],
                    "completion": completion,
                    "groundtruth": record["groundtruth"],
                    **scores,
                }
            )
            print(
                f"{record['metadata']['task_id']:<30} "
                f"exact_match={scores['exact_match']!s:<5} "
                f"edit_similarity={scores['edit_similarity']:.1f}"
            )
    finally:
        # Sauvegarder même en cas d'erreur (ex. rate limit) pour ne pas perdre
        # les résultats déjà générés.
        if results:
            save_jsonl(results, output_path)

    exact_matches = sum(r["exact_match"] for r in results)
    avg_edit_sim = sum(r["edit_similarity"] for r in results) / len(results)
    print()
    print(f"Résultats sauvegardés dans {output_path}")
    print(f"Exact match : {exact_matches}/{len(results)}")
    print(f"Edit similarity moyenne : {avg_edit_sim:.1f}")

    return results
