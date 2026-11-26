"""Module 1 support code: language codes, audio helpers, message types."""

from __future__ import annotations

import json

import numpy as np
import pytest

from shared import audio
from shared.languages import (
    FLORES_TO_ISO,
    ISO_TO_FLORES,
    UnsupportedLanguage,
    is_flores,
    normalize_lang,
    supported_languages,
    supports_tts,
    to_flores,
)
from shared.types import MAX_DATA_PACKET_BYTES, LatencyReport, SpeechSegment, SubtitleMessage


# --------------------------------------------------------------------- languages
def test_all_flores_codes_have_the_documented_shape():
    assert all(is_flores(c) for c in ISO_TO_FLORES.values())
    assert len(set(ISO_TO_FLORES.values())) == len(ISO_TO_FLORES)  # bijective
    assert FLORES_TO_ISO["eng_Latn"] == "en"


@pytest.mark.parametrize(
    "raw,expected",
    [("en", "en"), ("EN", "en"), ("en-US", "en"), ("zh_CN", "zh"), ("eng_Latn", "en"),
     ("deu_Latn", "de"), ("iw", "he"), ("nb", "no"), (" fr ", "fr")],
)
def test_normalize_lang(raw, expected):
    assert normalize_lang(raw) == expected


@pytest.mark.parametrize("bad", ["", "  ", "xx", "tlh", "abc_Latn", "klingon"])
def test_normalize_lang_rejects_unknown(bad):
    with pytest.raises(UnsupportedLanguage):
        normalize_lang(bad)


def test_to_flores_examples_from_model_card():
    assert to_flores("en") == "eng_Latn"
    assert to_flores("es") == "spa_Latn"
    assert to_flores("fr-FR") == "fra_Latn"
    assert to_flores("zh") == "zho_Hans"


def test_tts_support_matches_xtts_list():
    assert supports_tts("es") and supports_tts("zh-CN") and supports_tts("tr")
    assert not supports_tts("sv")  # not an XTTS-v2 language
    assert not supports_tts("klingon")


def test_supported_languages_sorted_and_nonempty():
    langs = supported_languages()
    assert langs == sorted(langs) and "de" in langs and len(langs) >= 30


# -------------------------------------------------------------------------- audio
def test_pcm_float_roundtrip_and_clipping():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0], dtype=np.float32)
    pcm = audio.float32_to_pcm16(x)
    assert pcm.dtype == np.int16
    assert pcm[3] == 32767 and pcm[4] == -32768 and pcm[5] == 32767  # clipped, not wrapped
    back = audio.pcm16_to_float32(pcm)
    assert np.allclose(back[:3], x[:3], atol=1 / 32768)


def test_pcm16_bytes_input():
    raw = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    out = audio.pcm16_to_float32(raw)
    assert np.allclose(out, [0.0, 0.5, -0.5])


def test_resample_changes_length_and_keeps_frequency():
    tone = audio.sine(440, 1.0, 16000)
    up = audio.resample(tone, 16000, 48000)
    assert len(up) == 48000
    spectrum = np.abs(np.fft.rfft(up))
    assert abs(np.argmax(spectrum) - 440) <= 1  # 1 s => 1 Hz bins
    assert audio.resample(tone, 16000, 16000) is not None and len(audio.resample(tone, 16000, 16000)) == 16000
    assert audio.resample(np.array([], dtype=np.float32), 16000, 8000).size == 0
    with pytest.raises(ValueError):
        audio.resample(tone, 0, 16000)


def test_to_mono_and_frames():
    stereo = np.stack([np.ones(10), -np.ones(10)], axis=1).astype(np.float32)
    assert np.allclose(audio.to_mono(stereo), 0.0)
    mono = np.arange(10, dtype=np.float32)
    assert audio.to_mono(mono) is mono
    got = list(audio.frames(mono, 4))
    assert [len(f) for f in got] == [4, 4]  # partial trailing frame dropped
    with pytest.raises(ValueError):
        list(audio.frames(mono, 0))
    with pytest.raises(ValueError):
        audio.to_mono(np.zeros((2, 2, 2)))


# --------------------------------------------------------------------------- types
def make_subtitle(**kw) -> SubtitleMessage:
    base = dict(segment_id="s1", speaker="alice", source_lang="de", text="Guten Tag",
                translations={"en": "Good day", "fr": "Bonne journée"}, is_final=True,
                start_time=1.2345, end_time=2.5)
    base.update(kw)
    return SubtitleMessage(**base)


def test_subtitle_json_roundtrip_with_unicode():
    msg = make_subtitle(text="Привет", translations={"ja": "こんにちは"})
    data = msg.to_bytes()
    assert "こんにちは".encode("utf-8") in data  # not \u-escaped
    back = SubtitleMessage.from_bytes(data)
    assert back.translations == {"ja": "こんにちは"} and back.text == "Привет" and back.is_final


def test_subtitle_payload_schema():
    obj = json.loads(make_subtitle().to_bytes())
    assert set(obj) == {"type", "version", "segment_id", "speaker", "source_lang", "text",
                        "translations", "is_final", "start_time", "end_time"}
    assert obj["type"] == "subtitle"
    assert obj["start_time"] in (1.234, 1.235)  # rounded to milliseconds


def test_subtitle_over_packet_limit_is_rejected():
    msg = make_subtitle(text="x" * (MAX_DATA_PACKET_BYTES + 1))
    with pytest.raises(ValueError, match="data-packet limit"):
        msg.to_bytes()


def test_subtitle_from_wrong_type_is_rejected():
    with pytest.raises(ValueError, match="not a subtitle"):
        SubtitleMessage.from_bytes(b'{"type":"chat"}')


def test_speech_segment_duration_and_latency_report():
    seg = SpeechSegment("s", "p", np.zeros(8000, dtype=np.float32), 16000, 0.0, 0.5)
    assert seg.duration == pytest.approx(0.5)
    rep = LatencyReport("s")
    rep.add("asr", 400.0)
    rep.add("nmt", 100.0)
    rep.add("asr", 50.0)
    assert rep.stages["asr"] == 450.0 and rep.total_ms == 550.0
