from app.schemas import Segment, TranscriptionJSON, TranscriptionVerboseJSON, Word


def test_transcription_json_serializes_text_only():
    payload = TranscriptionJSON(text="hello world")
    assert payload.model_dump() == {"text": "hello world"}


def test_transcription_verbose_json_serializes_all_fields():
    payload = TranscriptionVerboseJSON(
        text="hi",
        language="en",
        duration=1.5,
        segments=[Segment(id=0, start=0.0, end=1.5, text="hi")],
        words=[Word(word="hi", start=0.0, end=0.3)],
    )
    dumped = payload.model_dump()
    assert dumped["language"] == "en"
    assert dumped["segments"][0]["text"] == "hi"
    assert dumped["words"][0]["word"] == "hi"


def test_transcription_verbose_json_words_default_none():
    payload = TranscriptionVerboseJSON(
        text="hi", language="en", duration=1.0,
        segments=[Segment(id=0, start=0.0, end=1.0, text="hi")],
    )
    assert payload.words is None
