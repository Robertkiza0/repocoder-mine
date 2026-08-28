from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from .base import CodeCompletionResult, Example
from .dataset import load_cceval_dataset
from .llm import CompletionParams, VLLMClient
from .metrics import MetricCalculator
from .prompt import PromptBuilder, Tokenizer

logger = logging.getLogger(__name__)


async def perform_code_completion(
    example: Example,
    builder: PromptBuilder,
    params: CompletionParams,
    model: str,
    base_url: str,
) -> CodeCompletionResult:
    """CrossCodeEval 코드 완성 (프롬프트 구성 → 생성 → 메트릭 계산)."""
    # 프롬프트 구성
    prompt_build_result = builder.build(example)
    prompt = prompt_build_result.prompt

    # 생성
    response = await VLLMClient(base_url=base_url, model=model).acompletion(
        prompt, params
    )
    completions = [c["text"].strip() for c in response["choices"]]

    # 메트릭 계산
    edit_sims, id_f1s = MetricCalculator(
        lang=example.metadata.language
    ).compute_metrics(example, completions)

    return CodeCompletionResult(
        task_id=example.metadata.task_id,
        prompt_build_result=prompt_build_result,
        completions=completions,
        edit_sims=edit_sims,
        id_f1s=id_f1s,
        edit_sim=sum(edit_sims) / len(edit_sims),
        id_f1=sum(id_f1s) / len(id_f1s),
    )


T = TypeVar("T")


async def stream_tasks(
    coro_fns: list[Callable[[], Awaitable[T]]],
    *,
    max_concurrency: int | None = None,
    desc: str = "Processing",
    use_tqdm: bool = False,
    timeout: float | None = 1,
    return_when: str = asyncio.FIRST_COMPLETED,
) -> AsyncIterator[asyncio.Task[T]]:
    """슬라이딩 윈도우로 동시 실행 수를 제한하며 원래 순서대로 완료된 Task를 yield하는 async generator.

    Args:
        coro_fns: 인자가 바인딩된 no-arg async callable 리스트.
        max_concurrency: 동시 실행 제한 수. None이면 제한 없이 병렬 실행한다.
        desc: tqdm 진행바 설명.
        use_tqdm: True면 tqdm 진행바를 표시한다.
        timeout: asyncio.wait의 timeout (초).
        return_when: asyncio.wait의 return_when 조건.

    Yields:
        완료된 asyncio.Task (원래 순서). 호출 쪽에서 task.result()로 값을 꺼내거나 예외를 처리한다.
    """
    total = len(coro_fns)
    window = total if max_concurrency is None else min(max_concurrency, total)

    # 윈도우 크기만큼만 초기 task 생성
    task_to_idx: dict[asyncio.Task[T], int] = {}
    next_launch = 0
    for i in range(window):
        task = asyncio.create_task(coro_fns[i]())
        task_to_idx[task] = i
        next_launch = i + 1

    pending = set(task_to_idx)
    buffer: dict[int, asyncio.Task[T]] = {}
    next_yield = 0

    if use_tqdm:
        tbar_ctx = tqdm(total=total, desc=desc)
    else:
        tbar_ctx = nullcontext()

    with tbar_ctx as tbar:
        while pending or next_yield in buffer:
            if pending:
                done, pending = await asyncio.wait(
                    pending, timeout=timeout, return_when=return_when
                )
                for task in done:
                    idx = task_to_idx.pop(task)
                    buffer[idx] = task
                    # 완료된 슬롯에 다음 coroutine 투입
                    if next_launch < total:
                        new_task = asyncio.create_task(coro_fns[next_launch]())
                        task_to_idx[new_task] = next_launch
                        pending.add(new_task)
                        next_launch += 1
            # 버퍼에서 순서대로 drain
            while next_yield in buffer:
                yield buffer.pop(next_yield)
                next_yield += 1
                if tbar is not None:
                    tbar.update(1)


class ProgressTracker:
    """일정 간격마다 진행률, 속도, ETA를 로깅한다."""

    def __init__(self, total: int, log_interval: int = 20) -> None:
        self._total = total
        self._log_interval = log_interval
        self._completed = 0
        self._errors = 0
        self._start_time = time.time()
        self._next_log_at = log_interval

    @property
    def completed(self) -> int:
        """완료된 작업 수."""
        return self._completed

    def update(self, n: int = 1, errors: int = 0) -> None:
        """완료 수를 갱신하고, 간격에 도달하면 INFO 로그를 남긴다."""
        self._completed += n
        self._errors += errors
        if self._completed >= self._next_log_at or self._completed >= self._total:
            self._log()
            self._next_log_at = self._completed + self._log_interval

    def _log(self) -> None:
        """현재 진행상황을 INFO로 로깅한다."""
        elapsed = time.time() - self._start_time
        rate = self._completed / elapsed if elapsed > 0 else 0
        remaining = self._total - self._completed
        eta = remaining / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d (%.1f%%) | %.1f it/s | "
            "elapsed %.0fs | ETA %.0fs | errors %d",
            self._completed,
            self._total,
            self._completed / self._total * 100,
            rate,
            elapsed,
            eta,
            self._errors,
        )


async def main():
    """실험 메인 진입점."""
    parser = argparse.ArgumentParser(description="CCEval 코드 완성 실험")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8091/v1",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bigcode/starcoder",
        help="vLLM 서버 모델명",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "java", "csharp", "typescript"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "bm25",
            "unixcoder",
            "openai",
            "oracle_bm25",
            "oracle_unixcoder",
            "oracle_openai",
        ],
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4096,
        help="모델 최대 토큰 길이",
    )
    parser.add_argument(
        "--max-generation-tokens",
        type=int,
        default=64,
        help="생성 최대 토큰 수",
    )
    parser.add_argument(
        "--max-crossfile-tokens",
        type=int,
        default=3072,
        help="크로스파일 컨텍스트 토큰 예산",
    )
    parser.add_argument(
        "--fim",
        action="store_true",
        help="FIM 프롬프트 사용 여부",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n", type=int, default=8, help="completion 수")
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 출력 디렉토리 + 로깅
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_dir / "run.log")
    logging.basicConfig(
        handlers=[stream_handler, file_handler],
        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        level=logging.INFO,
    )

    # args 저장
    logger.info("Args: %s", args)
    args_text = json.dumps(vars(args), ensure_ascii=False, indent=2, default=str)
    (args.output_dir / "args.json").write_text(args_text)

    # 데이터셋 로드
    examples = load_cceval_dataset(
        language=args.language,
        split=args.split,
        sample=args.sample,
        seed=args.seed,
    )

    params = CompletionParams(
        max_tokens=args.max_generation_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n,
    )

    # 빌더(토크나이저 포함)는 한 번만 만들어 모든 example에서 재사용
    builder = PromptBuilder(
        tokenizer=Tokenizer(args.model_name),
        max_tokens=args.max_prompt_tokens,
        max_crossfile_tokens=args.max_crossfile_tokens,
        fim=args.fim,
    )

    coro_fns = [
        lambda ex=ex: perform_code_completion(
            example=ex,
            builder=builder,
            params=params,
            model=args.model_name,
            base_url=args.base_url,
        )
        for ex in examples
    ]

    completion_dir = args.output_dir / "completions"
    completion_dir.mkdir(exist_ok=True)

    tracker = ProgressTracker(total=len(coro_fns), log_interval=20)
    completed: dict[str, CodeCompletionResult] = {}
    async for task in stream_tasks(coro_fns, max_concurrency=args.max_concurrency):
        try:
            result = task.result()
        except Exception as e:
            logger.exception("Task failed: %s", e)
            tracker.update(errors=1)
            continue
        assert isinstance(result, CodeCompletionResult)
        completed[result.task_id] = result
        output_path = completion_dir / f"{result.task_id.replace('/', '__')}.json"
        output_path.write_text(result.model_dump_json(ensure_ascii=False, indent=2))
        tracker.update()

    # 통계 저장
    n_completed = len(completed)
    if n_completed > 0:
        edit_sim, id_f1 = 0, 0
        for r in completed.values():
            edit_sim += r.edit_sim
            id_f1 += r.id_f1
        edit_sim /= n_completed
        id_f1 /= n_completed
    else:
        edit_sim = id_f1 = 0

    metrics = {
        "edit_sim": round(edit_sim, 2),
        "id_f1": round(id_f1, 2),
        "n_examples": len(examples),
        "n_completed": n_completed,
        "errors": len(examples) - n_completed,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    logger.info("Done: %s", metrics)


if __name__ == "__main__":
    asyncio.run(main())
