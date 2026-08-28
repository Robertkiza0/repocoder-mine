# repocoder-mine

A minimal reimplementation of [RepoCoder](https://arxiv.org/abs/2303.12570)'s
iterative retrieval-generation pipeline for repository-level code completion,
evaluated on [CrossCodeEval (CCEval)](https://github.com/amazon-science/cceval).

## How it works

For each line-completion task in CCEval, the pipeline runs two
retrieval-generation iterations:

1. **Query** — built from the tail of the unfinished code (`S_w` lines on the
   first iteration; on later iterations, `S_w - S_s` lines of the unfinished
   code concatenated with the first `S_s` lines of the previous prediction,
   so the model's own draft steers the next retrieval).
2. **Retrieve** — the top-k most similar code snippets from the same
   repository, ranked by Jaccard similarity over a bag-of-words tokenization.
3. **Prompt** — the retrieved snippets (ascending score order, closest match
   last, each labeled with its file path) are placed before the unfinished
   code.
4. **Generate** — an LLM (Groq, `openai/gpt-oss-20b`) completes the prompt.

The repository's own code is pre-split into overlapping snippets using
RepoCoder's default sliding-window hyperparameters: window size `S_w = 20`
lines, stride `S_s = 10` lines.

## Project layout

| File | Role |
|---|---|
| `dataset.py` | Load/save CCEval JSONL, sliding-window extraction per repository, `last_lines`/`first_lines` helpers |
| `retriever.py` | Jaccard bag-of-words retriever (`retrieve_top_k_from_repository`); also a TF-IDF cosine baseline (`best_snippet`) |
| `prompt_builder.py` | Formats retrieved chunks + unfinished code into the final LLM prompt |
| `iterate.py` | The iterative retrieval-generation loop (`run_repocoder_pipeline`) — takes any `generate` callable, independent of the LLM backend |
| `generator.py` | **Orchestrator** — calls the Groq API (`call_model_api`), and `run_experiment` (dataset → retrieval → generation → evaluation) |
| `run_completion.py` | **CLI entry point** — argparse wrapper around `generator.run_experiment` (mirrors `cceval/run_completion.py`'s `-o/--output-dir`, `args.json`, `metrics.json` conventions) |
| `metrics.py` | `exact_match`, `edit_similarity` (thefuzz), plus the official CCEval `edit_similarity` / `identifier_f1` via `cceval.metrics` |
| `chunker.py` | Standalone line-based chunker (early prototype, not used by the current pipeline) |
| `cceval/` | Vendored copy of [CrossCodeEval](https://github.com/amazon-science/cceval)'s evaluation code (`metrics.py` needs it importable) |
| `data/repositories/` | The pre-built per-repository snippet database (471 files, one per repo) — see "How it works" |

Dependency direction: `dataset` → `retriever` / `prompt_builder` → `iterate` →
`generator`. No circular imports.

## Setup

```bash
pip install -r requirements.txt
```

`--backend huggingface` additionally needs `torch`, `transformers`, and
`accelerate` (commented out in `requirements.txt` — heavy, install only if
you use that backend).

Copy `.env.example` to `.env` and set your key (only needed for
`--backend groq`, the default):

```
GROQ_API_KEY=your_key_here
```

`.env` is git-ignored.

You'll also need CCEval's raw `line_completion.jsonl` (not included here —
download it from the [CrossCodeEval dataset](https://github.com/amazon-science/cceval))
to run new experiments; `data/repositories/` was already built from it.

## Usage

### 1. (Optional) Rebuild the per-repository snippet database

Only needed if you want to regenerate `data/repositories/` from a fresh
`line_completion.jsonl` (already included in this repo otherwise):

```bash
python dataset.py
```

Splits `line_completion.jsonl` into train/validation/test, and extracts
sliding-window snippets grouped by repository into `data/repositories/`
(one `.jsonl` file per repository).

### 2. Run the experiment

```bash
python run_completion.py -o results/run1 --sample 20
```

Loads `line_completion.jsonl`, runs the iterative RepoCoder pipeline on each
example, evaluates each completion, and writes `results.jsonl`,
`metrics.json` (averaged scores) and `args.json` (the run's parameters) into
the output directory.

Options: `--backend {groq,ollama,huggingface}`, `--model`, `--line-completion-path`,
`--repositories-dir`, `--k` (retrieved chunks), `--iterations` (RepoCoder
rounds), `--sample` (`0` = the full 2665-example dataset).

```bash
# Local model via Ollama (must be running: `ollama pull starcoder2:3b`)
python run_completion.py -o results/run-ollama --backend ollama --sample 5

# Local model via transformers (e.g. on a Colab GPU)
python run_completion.py -o results/run-hf --backend huggingface \
    --model bigcode/starcoderbase-7b --sample 20
```

Results are saved incrementally on failure (e.g. a rate limit), so a crash
mid-run doesn't lose already-generated completions.

## Known limitations

- **`exact_match` is almost always `False`**: `call_model_api` returns the
  raw multi-line continuation, while CCEval's `groundtruth` is a single
  line. `edit_similarity` and `identifier_f1` (from `cceval.metrics`,
  post-processed to the first completed statement) are the meaningful
  metrics here; truncating the raw completion to its first line before
  comparing would make `exact_match` meaningful too.
- **Groq free tier** has both a per-minute and a per-day token limit —
  large `SAMPLE_SIZE` runs can hit `RateLimitError` (429) partway through.
- The Jaccard retriever can only surface code that already exists elsewhere
  in the same repository — it cannot help when the target line is genuinely
  novel.
