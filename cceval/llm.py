"""vLLM 백엔드 — OpenAI 호환 엔드포인트 클라이언트 (chat/completions)."""

from __future__ import annotations

from functools import lru_cache

import openai
from pydantic import BaseModel


class CompletionParams(BaseModel):
    """vLLM /completions 생성 파라미터 (model은 VLLMClient가 보유한다)."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    n: int | None = None
    echo: bool | None = None
    logprobs: int | None = None


GREEDY_COMPLETION_PARAMS = CompletionParams(temperature=0.0)  # greedy decoding 기본값


class VLLMClient:
    """vLLM /completions 클라이언트 (OpenAI SDK 기반).

    base_url별로 SDK 클라이언트(httpx 커넥션 풀 + 재시도 내장)를 재사용한다.
    top_k 등 OpenAI 비표준 파라미터는 extra_body로 전달해 요청 본문을 vLLM raw 요청과 동일하게 만든다.
    응답은 model_dump()로 dict로 변환해 반환한다.
    api_key가 없으면 vLLM용 placeholder("EMPTY")를 쓴다 (OpenAI 등 호환 엔드포인트는 실제 키 전달).
    chat completions은 ChatClient가 담당한다.

    주의: SDK 비동기 클라이언트는 생성된 이벤트 루프에 묶이므로, 한 프로세스에서
    여러 asyncio.run()에 걸쳐 재사용하지 않는다 (실험 스크립트는 단일 run).
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    async def acompletion(
        self, prompt: str, params: CompletionParams = GREEDY_COMPLETION_PARAMS
    ) -> dict:
        """비동기 completions 요청을 보낸다 (기본 greedy decoding)."""
        client = async_client(self.base_url, self.api_key, self.timeout)
        response = await client.completions.create(
            model=self.model,
            prompt=prompt,
            extra_body=params.model_dump(exclude_none=True),
        )
        return response.model_dump()


# ── SDK 클라이언트 팩토리 (base_url별 재사용) ──────────────────────


@lru_cache(maxsize=None)
def async_client(base_url: str, api_key: str, timeout: float) -> openai.AsyncOpenAI:
    """base_url별 비동기 SDK 클라이언트를 재사용한다 (httpx 커넥션 풀 + 재시도)."""
    return openai.AsyncOpenAI(
        base_url=base_url, api_key=api_key, timeout=timeout, max_retries=3
    )


@lru_cache(maxsize=None)
def sync_client(base_url: str, api_key: str, timeout: float) -> openai.OpenAI:
    """base_url별 동기 SDK 클라이언트를 재사용한다 (httpx 커넥션 풀 + 재시도)."""
    return openai.OpenAI(
        base_url=base_url, api_key=api_key, timeout=timeout, max_retries=3
    )
