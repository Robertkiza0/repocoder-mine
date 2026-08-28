# CCEval

A repo-level code completion experiment tool based on [CrossCodeEval](https://github.com/amazon-science/cceval).
It sends prompts to a vLLM (OpenAI-compatible) server to generate completions, then computes metrics such as Exact Match and Edit Similarity.

> **Where to run:** Run every command from the **repository root**.
> - `cceval` is a package that uses relative imports, so it must be launched via `python -m cceval.run_completion` (running `python run_completion.py` inside `cceval/` will not work).
> - Both the vLLM launch script and `run_completion` load the `.env` at the repo root, so running them from the same location keeps the environment variables consistent.

## 1. Setup

```bash
conda create -n cceval python=3.12
conda activate cceval
pip install -r cceval/requirements.txt
```

## 2. Configure `.env`

Place a `.env` at the repository root and fill in the following values.

```bash
# Dataset (required) — CrossCodeEval data root
CCEVAL_DATA_DIR=<<<cceval-data-dir>>>

# For the vLLM server
CACHE_DIR=<<<root for the huggingface cache>>>   # required (used as <CACHE_DIR>/hf when HF_HOME is unset)
HF_TOKEN=<<<huggingface-token>>>                 # for gated model access
DEVICES=3                                         # GPU device id to use (default "3")
```

## 3. Dataset layout

Under `CCEVAL_DATA_DIR`, data is organized per language.

```
<CCEVAL_DATA_DIR>/
├── python
│   ├── line_completion.jsonl                              # baseline
│   ├── line_completion_rg1_bm25.jsonl                     # bm25
│   ├── line_completion_rg1_unixcoder_cosine_sim.jsonl     # unixcoder
│   ├── line_completion_rg1_openai_cosine_sim.jsonl        # openai
│   ├── line_completion_oracle_bm25.jsonl                  # oracle_bm25
│   ├── line_completion_oracle_unixcoder_cosine_sim.jsonl  # oracle_unixcoder
│   └── line_completion_oracle_openai_cosine_sim.jsonl     # oracle_openai
├── java
├── csharp
└── typescript
```

Every language directory has the same set of files. The `--split` value maps to the filename shown in the comments above.

## 4. Launch the vLLM server

Serve the model that generates completions with a vLLM OpenAI-compatible server. The script loads `.env` and runs the vLLM container via Docker, using the `DEVICES` value from `.env` to select the GPU.

```bash
bash cceval/launch-vllm-server/run-starcoder.sh
```

- Serves `bigcode/starcoder` on port `8091`.
- Defaults align with `--base-url http://localhost:8091/v1` and `--model-name bigcode/starcoder`.
- To use a different model/port, copy and edit the script, and override `--base-url` / `--model-name` at run time.

## 5. Run completion

```bash
python -m cceval.run_completion -o results/starcoder/baseline
```

Results are written to the directory given by `-o`.

### Sanity check

Running the default configuration (`bigcode/starcoder`, `--language python`, `--split baseline`) reproduces the following, which closely matches the StarCoder-15.5B in-file (no cross-file context) numbers reported in the CrossCodeEval paper (Table 2). Use it to confirm the pipeline runs end to end.

| Metric | Reported (paper) | Reproduced |
|--------|:----------------:|:----------:|
| Edit Similarity (`edit_sim`) | 61.08 | 60.72 |
| Identifier F1 (`id_f1`) | 48.16 | 46.54 |

Reproduced run: `{'edit_sim': 60.72, 'id_f1': 46.54, 'n_examples': 2665, 'n_completed': 2665, 'errors': 0}` — all 2665 examples completed with no server errors.

> Reported numbers: Ding et al., *CrossCodeEval* (NeurIPS 2023), Table 2 — StarCoder-15.5B, Python, no cross-file context. Small gaps are expected since the prompt construction and token budget differ from the paper.

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | (required) | Output directory for results |
| `--base-url` | `http://localhost:8091/v1` | vLLM server URL |
| `--model-name` | `bigcode/starcoder` | Model name on the vLLM server |
| `--language` | `python` | `python` / `java` / `csharp` / `typescript` |
| `--split` | `baseline` | `baseline` / `bm25` / `unixcoder` / `openai` / `oracle_bm25` / `oracle_unixcoder` / `oracle_openai` |
| `--max-prompt-tokens` | `4096` | Max prompt token length |
| `--max-generation-tokens` | `64` | Max tokens to generate |
| `--max-crossfile-tokens` | `3072` | Cross-file context token budget |
| `--fim` | off | Use FIM (Fill-In-the-Middle) prompts |
| `--temperature` | `0.2` | Sampling temperature |
| `--top-p` | `0.95` | Nucleus sampling |
| `--top-k` | `20` | Top-k sampling |
| `--n` | `8` | Completions per example |
| `--max-concurrency` | `64` | Concurrent requests |
| `--sample` | full | Number of examples to randomly sample from the dataset |
| `--seed` | `42` | Seed for `--sample` sampling |

### Example

```bash
# Java, BM25 retrieval context, sampling only 100 examples
python -m cceval.run_completion \
    -o results/starcoder/java-bm25 \
    --language java \
    --split bm25 \
    --sample 100
```
