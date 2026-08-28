"""CrossCodeEval 프롬프트 구성 — in-file/cross-file 컨텍스트를 토큰 예산에 맞춰 조립한다."""

from transformers import AutoTokenizer

from .base import CodeChunk, Example, PromptBuildResult

FIM_TOKENS = {
    "bigcode/starcoder": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "bigcode/starcoder2-7b": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "bigcode/starcoderbase-7b": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "Qwen/Qwen2.5-Coder-7B": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
    "xiaowu0162/repoformer-16b": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
}


def create_fim_prompt(left_context: str, right_context: str, model_name: str) -> str:
    fim_prefix, fim_suffix, fim_middle = FIM_TOKENS[model_name]
    return fim_prefix + left_context + fim_suffix + right_context + fim_middle


class Tokenizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def tokenize(self, text: str) -> list[int]:
        return self.hf_tokenizer.encode(text, add_special_tokens=False)


class PromptBuilder:
    """CCEval completion 프롬프트 빌더.

    실험 전체에서 공유되는 설정(tokenizer, 토큰 예산, FIM 여부)을 인스턴스 상태로 보관하고,
    `build`로 `Example`마다 in-file/cross-file 컨텍스트를 예산에 맞춰 조립한다.
    빌더는 immutable하게 재사용된다 — example별로 변하는 값(language 등)만 메서드로 전달한다.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_tokens: int,
        max_crossfile_tokens: int,
        fim: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_crossfile_tokens = max_crossfile_tokens
        self.fim = fim

    def build(self, example: Example) -> PromptBuildResult:
        """`Example`로 completion 프롬프트를 구성한다.

        example.prompt/right_context는 cursor 앞/뒤 in-file 컨텍스트, crossfile_context는
        cross-file 후보(앞에 붙는다). cross-file 예산이 0이거나 chunk가 없으면 in-file만으로,
        아니면 cross-file 블록을 앞에 붙여 구성한다. 프롬프트에 실제로 포함된 cross-file
        정보도 함께 반환한다.
        """
        if self.max_crossfile_tokens == 0:
            return self._build_local(example)
        if example.crossfile_context is None:
            return self._build_local(example)
        if len(example.crossfile_context.list) == 0:
            return self._build_local(example)
        return self._build_augmented(example)

    def _build_local(self, example: Example) -> PromptBuildResult:
        """in-file context만으로 프롬프트를 구성한다 (cross-file 없음). truncate를 고려한다."""
        left = example.prompt
        right = example.right_context
        left_lines = left.split("\n")
        right_lines = right.split("\n")
        left, right, num_infile_lines = self._fit_infile(left_lines, right_lines)
        prompt = self._create_final_prompt(left, right)
        return PromptBuildResult(
            prompt=prompt,
            crossfile_context=[],
            num_infile_lines=num_infile_lines,
            infile_context=(left, right),
        )

    def _build_augmented(self, example: Example) -> PromptBuildResult:
        """cross-file 블록을 앞에 붙인 프롬프트를 구성한다 (chunk가 있다고 가정).

        cross-file chunk를 예산에 맞춰 고르고(`_fit_crossfile`) verbalize한 뒤, 그 블록을 앞에
        두고 in-file을 예산에 맞춰 붙인다(`_fit_infile`). 프롬프트에 실제로 포함된 cross-file
        정보도 함께 반환한다.
        """
        left = example.prompt
        right = example.right_context
        language = example.metadata.language
        code_chunks = (
            example.crossfile_context.list if example.crossfile_context else []
        )
        effective_code_chunk, num_chunks, num_crossfile_last_chunk_lines = (
            self._fit_crossfile(code_chunks, language)
        )
        crossfile_prompt = self._format_code_chunks(effective_code_chunk, language)

        left_lines = left.split("\n")
        right_lines = right.split("\n") if self.fim else []
        left, right, num_infile_lines = self._fit_infile(
            left_lines, right_lines, crossfile_prompt
        )
        prompt = self._create_final_prompt(left, right, crossfile_prompt)
        return PromptBuildResult(
            prompt=prompt,
            num_infile_lines=num_infile_lines,
            infile_context=(left, right),
            num_crossfile_chunks=num_chunks,
            num_crossfile_last_chunk_lines=num_crossfile_last_chunk_lines,
            crossfile_context=effective_code_chunk,
        )

    # ── 토큰 예산 맞춤 ────────────────────────────────────────────

    def _fit_crossfile(
        self, code_chunks: list[CodeChunk], lang: str
    ) -> tuple[list[CodeChunk], int, int | None]:
        """토큰 예산에 맞는 cross-file chunk 집합을 고른다.

        먼저 chunk-level로 통째로 들어갈 chunk 수를 정하고, 자리가 남으면 다음 chunk 하나를
        line-level로 잘라 마저 채운다. 반환: (effective_chunks, num_chunks, num_last_chunk_lines).
        """

        def chunk_fits(m: int) -> bool:
            candidate = self._format_code_chunks(
                self._truncate_code_chunks(code_chunks, m), lang
            )
            return len(self.tokenizer.tokenize(candidate)) < self.max_crossfile_tokens

        num_chunks = self._largest_fitting(len(code_chunks) + 1, chunk_fits)

        num_last_chunk_lines = None
        if num_chunks < len(code_chunks):
            # line-level: 다음 chunk를 줄 단위로 잘라 남은 예산을 채운다.
            num_chunks += 1
            last_total = len(code_chunks[num_chunks - 1].retrieved_chunk.splitlines())

            def line_fits(m: int) -> bool:
                candidate = self._format_code_chunks(
                    self._truncate_code_chunks(code_chunks, num_chunks, m), lang
                )
                return (
                    len(self.tokenizer.tokenize(candidate)) < self.max_crossfile_tokens
                )

            num_last_chunk_lines = self._largest_fitting(last_total, line_fits)

        effective = self._truncate_code_chunks(
            code_chunks, num_chunks, num_last_chunk_lines
        )
        return effective, num_chunks, num_last_chunk_lines

    def _fit_infile(
        self,
        left_lines: list[str],
        right_lines: list[str],
        crossfile_prompt: str | None = None,
    ) -> tuple[str, str, int]:
        """in-file context를 토큰 예산에 맞게 라인 단위로 truncate한다.

        반환: (left, right, num_infile_lines).
        """

        def fits(m: int) -> bool:
            left, right = self._truncate_infile_context(left_lines, right_lines, m)
            candidate = self._create_final_prompt(left, right, crossfile_prompt)
            return len(self.tokenizer.tokenize(candidate)) < self.max_tokens

        num_infile_lines = self._largest_fitting(
            max(len(left_lines), len(right_lines)) + 1, fits
        )
        left, right = self._truncate_infile_context(
            left_lines, right_lines, num_infile_lines
        )
        return left, right, num_infile_lines

    def _create_final_prompt(
        self, left: str, right: str, crossfile_prompt: str | None = None
    ) -> str:
        prompt = left if crossfile_prompt is None else crossfile_prompt + left
        if self.fim:
            prompt = create_fim_prompt(prompt, right, self.tokenizer.model_name)
        return prompt

    def _truncate_infile_context(
        self, left_lines: list[str], right_lines: list[str], num_lines: int
    ) -> tuple[str, str]:
        left = "\n".join(left_lines[len(left_lines) - num_lines :])
        right = "\n".join(right_lines[:num_lines]) if self.fim else ""
        return left, right

    # ── 포맷팅 · 저수준 유틸 (상태 없음) ──────────────────────────

    @staticmethod
    def _format_code_chunks(code_chunks: list[CodeChunk], lang: str) -> str:
        """retrieved code chunk들을 CCEval 표준 cross-file 블록으로 직렬화한다 (truncation 무관)."""
        prompt = (
            PromptBuilder._prepend_comment(
                "Here are some relevant code fragments from other files of the repo:",
                lang,
            )
            + "\n\n"
        )
        for code_chunk in code_chunks:
            prompt += PromptBuilder._format_code_chunk(code_chunk, lang) + "\n\n"
        return prompt

    @staticmethod
    def _format_code_chunk(code_chunk: CodeChunk, lang: str) -> str:
        content = code_chunk.retrieved_chunk.strip()
        text = "The below code fragment can be found in:\n{filepath}\n{content}".format(
            filepath=code_chunk.filename, content=content
        )
        return PromptBuilder._prepend_comment(text, lang)

    @staticmethod
    def _prepend_comment(text: str, lang: str) -> str:
        comment_prefix = "# " if lang == "python" else "// "
        return "\n".join(f"{comment_prefix}{line}" for line in text.strip().split("\n"))

    @staticmethod
    def _truncate_code_chunks(
        code_chunks: list[CodeChunk],
        num_chunks: int,
        num_last_chunk_lines: int | None = None,
    ) -> list[CodeChunk]:
        """앞 num_chunks개만 취하고, num_last_chunk_lines가 주어지면 마지막 chunk를 그 줄 수로 자른다."""
        truncated_chunks = []
        for i, chunk in enumerate(code_chunks[:num_chunks]):
            if i == num_chunks - 1 and num_last_chunk_lines is not None:
                truncated_chunks.append(
                    CodeChunk(
                        filename=chunk.filename,
                        retrieved_chunk="\n".join(
                            chunk.retrieved_chunk.splitlines()[:num_last_chunk_lines]
                        ),
                        score=chunk.score,
                    )
                )
            else:
                truncated_chunks.append(chunk)
        return truncated_chunks

    @staticmethod
    def _largest_fitting(hi: int, fits) -> int:
        """[0, hi) 에서 fits(n)이 참인 최대 n을 binary search로 찾는다.

        fits는 n에 대해 단조(n이 작을수록 잘 맞음)라고 가정한다. 모두 안 맞으면 0을 반환.
        """
        s, e = 0, hi
        while (e - s) > 1:
            m = (s + e) // 2
            s, e = (m, e) if fits(m) else (s, m)
        return s
