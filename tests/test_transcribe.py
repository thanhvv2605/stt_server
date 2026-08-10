import threading
import time

from app.transcribe import TranscriptionService


class FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class FakeSegment:
    def __init__(self, id, start, end, text, words=None):
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class FakeInfo:
    def __init__(self, language, duration):
        self.language = language
        self.duration = duration


class FakeWhisperModel:
    def __init__(self, segments, language="en", duration=1.0, delay=0.0):
        self._segments = segments
        self._language = language
        self._duration = duration
        self._delay = delay
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = []

    def transcribe(self, audio, language=None, word_timestamps=False):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append((audio, language, word_timestamps))
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.in_flight -= 1
        return iter(self._segments), FakeInfo(self._language, self._duration)


async def test_transcribe_returns_joined_text_and_segments():
    segments = [FakeSegment(0, 0.0, 1.0, " hello"), FakeSegment(1, 1.0, 2.0, " world")]
    model = FakeWhisperModel(segments, language="en", duration=2.0)
    service = TranscriptionService(model, max_concurrent=2)

    result = await service.transcribe("audio.wav", language=None, word_timestamps=False)

    assert result.text == "hello world"
    assert result.language == "en"
    assert result.duration == 2.0
    assert [s.text for s in result.segments] == [" hello", " world"]
    service.shutdown()


async def test_transcribe_includes_word_timestamps_when_requested():
    words = [FakeWord("hi", 0.0, 0.3)]
    segments = [FakeSegment(0, 0.0, 1.0, " hi", words=words)]
    model = FakeWhisperModel(segments)
    service = TranscriptionService(model, max_concurrent=2)

    result = await service.transcribe("audio.wav", language=None, word_timestamps=True)

    assert result.segments[0].words[0].word == "hi"
    service.shutdown()


async def test_transcribe_passes_language_through():
    segments = [FakeSegment(0, 0.0, 1.0, " hi")]
    model = FakeWhisperModel(segments)
    service = TranscriptionService(model, max_concurrent=2)

    await service.transcribe("audio.wav", language="vi", word_timestamps=False)

    assert model.calls[0] == ("audio.wav", "vi", False)
    service.shutdown()


async def test_transcribe_limits_concurrency_to_max_concurrent():
    import asyncio

    segments = [FakeSegment(0, 0.0, 1.0, " hi")]
    model = FakeWhisperModel(segments, delay=0.2)
    service = TranscriptionService(model, max_concurrent=2)

    await asyncio.gather(*(
        service.transcribe("audio.wav", language=None, word_timestamps=False)
        for _ in range(5)
    ))

    assert model.max_in_flight == 2
    service.shutdown()
