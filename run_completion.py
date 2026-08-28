"""Point d'entrée CLI — lance l'expérimentation RepoCoder complète.

Reprend les conventions de cceval/run_completion.py (-o/--output-dir,
args.json, metrics.json) pour ce pipeline Jaccard + Groq.
"""

import argparse
import functools
import json
from pathlib import Path

from generator import call_huggingface_api, call_model_api, call_ollama_api, run_experiment


def main() -> None:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="RepoCoder mine — expérimentation complète")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--line-completion-path",
        type=Path,
        default=Path(r"C:\Users\User\Downloads\line_completion.jsonl"),
        help="chemin vers line_completion.jsonl",
    )
    parser.add_argument(
        "--repositories-dir",
        type=Path,
        default=project_dir / "data" / "repositories",
        help="dossier des blocs de code par dépôt (dataset.save_repository_snippets)",
    )
    parser.add_argument(
        "--backend",
        choices=["groq", "ollama", "huggingface"],
        default="groq",
        help="groq (API distante), ollama (modèle local) ou huggingface (modèle local, transformers)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "nom du modèle (défaut : openai/gpt-oss-20b pour groq, "
            "starcoder2:3b pour ollama, bigcode/starcoderbase-7b pour huggingface)"
        ),
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434",
        help="URL du serveur Ollama local (--backend ollama)",
    )
    parser.add_argument("--k", type=int, default=10, help="nombre d'extraits récupérés (Jaccard)")
    parser.add_argument("--iterations", type=int, default=2, help="itérations RepoCoder")
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="nombre d'exemples à évaluer (0 = tout le dataset)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args_text = json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2)
    (args.output_dir / "args.json").write_text(args_text)

    if args.backend == "ollama":
        model = args.model or "starcoder2:3b"
        generate = functools.partial(call_ollama_api, model=model, base_url=args.ollama_base_url)
    elif args.backend == "huggingface":
        model = args.model or "bigcode/starcoderbase-7b"
        generate = functools.partial(call_huggingface_api, model=model)
    else:
        model = args.model or "openai/gpt-oss-20b"
        generate = functools.partial(call_model_api, model=model)

    sample_size = None if args.sample == 0 else args.sample

    results = run_experiment(
        line_completion_path=args.line_completion_path,
        repositories_dir=args.repositories_dir,
        output_path=args.output_dir / "results.jsonl",
        k=args.k,
        iterations=args.iterations,
        sample_size=sample_size,
        generate=generate,
    )

    if not results:
        return

    metrics = {
        "exact_match": sum(r["exact_match"] for r in results) / len(results),
        "edit_similarity": sum(r["edit_similarity"] for r in results) / len(results),
        "identifier_f1": sum(r["identifier_f1"] for r in results) / len(results),
        "n_completed": len(results),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
