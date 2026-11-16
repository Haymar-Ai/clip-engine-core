"""Shared pytest setup. `slow` tests (a real whisper model / real media tools) are opt-in with
`-m slow`. No test reaches the network, an LLM or a real whisper model: every step that would is
off for the suite, and the tests that cover such a step turn it on with fakes."""
import os
import tempfile

import pytest

# the engine reads its config on import: point the scratch dir at a temp folder first, so no test
# writes caches into the repository
os.environ.setdefault("CLIPENGINE_SCRATCH_DIR", tempfile.mkdtemp(prefix="clipengine-tests-"))
# the heavy-whisper slots are host-wide flock files: tests take their own, never a real run's
os.environ.setdefault("HEAVY_WHISPER_LOCK_DIR", tempfile.mkdtemp(prefix="clipengine-slots-"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads a real model or runs real media tools; "
                                       "skipped unless selected with -m slow")
    # the cut review never calls a real LLM from a test, nor reads audio for the cut seat
    os.environ["CUT_REVIEW"] = "0"
    os.environ["CUT_SEAT"] = "0"
    # nor the topic map's outline call; test_topic_map turns it on with a fake LLM
    os.environ["TOPIC_MAP"] = "0"
    # nor the final cut: it reads the window audio and the second ASR; test_final_cut turns it on
    # with fakes
    os.environ["FINAL_CUT"] = "0"
    # nor the caption check: it runs the second ASR and the strong model
    os.environ["CAPTION_CHECK"] = "0"
    # nor do the cut checks read the source audio (quality_gate.cut_audio, an ffmpeg read many
    # render tests count or fake); test_gate_audio_checks turns it on
    os.environ["CUT_AUDIO"] = "0"
    # nor does a failed caption check run the tie-break model (whisper medium on the source)
    os.environ["CAPTION_TIEBREAK"] = "0"
    # nor does select hear the whole window with the strong model; test_window_pass turns it on
    # with fakes
    os.environ["WINDOW_TRANSCRIPT"] = "0"
    # nor does the render check crosstalk on the second ASR; test_gate_audio_checks turns it on with fakes
    os.environ["CROSSTALK_CONFIRM"] = "0"


def pytest_collection_modifyitems(config, items):
    if "slow" in (config.getoption("markexpr", "") or ""):
        return                  # -m names slow: either selects them or already deselects them
    skip = pytest.mark.skip(reason="slow test: opt in with -m slow")
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_runstate():
    """Every test starts with no sources, clips or events."""
    from clipengine import runstate
    runstate.reset()
    yield
    runstate.reset()
