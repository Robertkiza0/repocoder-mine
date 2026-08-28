"""CrossCodeEval 관련 데이터 모델."""

from pydantic import BaseModel


class MetaData(BaseModel):
    """CCEval 태스크 메타데이터."""

    task_id: str
    repository: str
    file: str

    @property
    def language(self) -> str:
        """task_id에서 언어를 추출한다. (e.g. 'project_cc_python/62' → 'python')"""
        prefix = self.task_id.split("/")[0]  # "project_cc_python"
        return prefix.split("_", 2)[-1]  # "python"


class CodeChunk(BaseModel):
    """retrieval 결과 파일 조각."""

    filename: str
    retrieved_chunk: str
    score: float


class CrossFileContext(BaseModel):
    """CCEval 크로스파일 컨텍스트."""

    text: str
    list: list[CodeChunk]


class Example(BaseModel):
    """CCEval 예제."""

    metadata: MetaData
    prompt: str
    groundtruth: str
    right_context: str
    crossfile_context: CrossFileContext | None = None


class PromptBuildResult(BaseModel):
    prompt: str
    num_infile_lines: int
    infile_context: tuple[str, str]  # (left_context, right_context)
    num_crossfile_chunks: int | None = None
    num_crossfile_last_chunk_lines: int | None = None
    crossfile_context: list[CodeChunk] | None


class CodeCompletionResult(BaseModel):
    task_id: str
    prompt_build_result: PromptBuildResult
    completions: list[str]
    edit_sims: list[float]
    id_f1s: list[float]
    edit_sim: float
    id_f1: float
