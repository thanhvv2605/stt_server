import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WordResult:
    word: str
    start: float
    end: float


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[WordResult]] = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: List[SegmentResult]


class TranscriptionService:
    def __init__(self, model, max_concurrent: int):
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._waiting = 0

    @property
    def queue_depth(self) -> int:
        return self._waiting

    async def transcribe(
        self, audio_path: str, language: Optional[str], word_timestamps: bool
    ) -> TranscriptionResult:
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._transcribe_sync, audio_path, language, word_timestamps
            )
        finally:
            self._semaphore.release()

    def _transcribe_sync(
        self, audio_path: str, language: Optional[str], word_timestamps: bool
    ) -> TranscriptionResult:
        segments_iter, info = self._model.transcribe(
            audio_path, language=language, word_timestamps=word_timestamps
        )
        segments: List[SegmentResult] = []
        text_parts: List[str] = []
        for i, seg in enumerate(segments_iter):
            words = None
            if word_timestamps and getattr(seg, "words", None):
                words = [WordResult(word=w.word, start=w.start, end=w.end) for w in seg.words]
            segments.append(SegmentResult(id=i, start=seg.start, end=seg.end, text=seg.text, words=words))
            text_parts.append(seg.text)
        return TranscriptionResult(
            text="".join(text_parts).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
