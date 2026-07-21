"""Shared pytest fixtures for the CrisperWhisper package tests.

The pure-Python unit tests (Viterbi alignment, prompt building, longform
chunk math, audio helpers, attention-stitching invariants) run anywhere
with no model or GPU.

The end-to-end tests need a model and a CUDA device.  Point them at models
via environment variables; when unset, the dependent tests are **skipped**
(never failed):

    CW2_MODEL_PATH        CTranslate2 main model (a dir with model.bin)
    CW2_DRAFT_MODEL_PATH  CTranslate2 draft model for speculative decoding
    CW2_HF_MODEL_PATH     HuggingFace model id/dir for the transformers
                          backend (e.g. a CrisperWhisper v2 HF checkpoint)

Audio fixtures use the checked-in ``samples/`` directory, so they need no
external data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MAIN_MODEL_ENV = "CW2_MODEL_PATH"
DRAFT_MODEL_ENV = "CW2_DRAFT_MODEL_PATH"
HF_MODEL_ENV = "CW2_HF_MODEL_PATH"
V1_MODEL_ENV = "CW_V1_MODEL_PATH"


def _find_samples_dir() -> Path | None:
    """Locate the ``samples/`` audio directory by walking up from this file.

    The package ships a self-contained ``samples/`` directory at its root, so
    the tests run with no external data.  Walking up the parents keeps this
    robust to layout (``crisperwhisper/tests/`` inside the monorepo or
    ``tests/`` in the standalone repo).  Honours ``CW2_SAMPLES_DIR`` override.
    """
    override = os.environ.get("CW2_SAMPLES_DIR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "samples"
        if candidate.is_dir():
            return candidate
    return None


_SAMPLES_DIR = _find_samples_dir()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: end-to-end test that requires a converted CT2 model + CUDA "
        "(skipped unless CW2_MODEL_PATH is set).",
    )
    config.addinivalue_line(
        "markers",
        "transformers: end-to-end test on the HuggingFace transformers "
        "backend (skipped unless CW2_HF_MODEL_PATH is set and torch is "
        "installed).",
    )
    config.addinivalue_line(
        "markers",
        "legacy: end-to-end test of the deprecated v1 model on the "
        "transformers backend (skipped unless CW_V1_MODEL_PATH is set and "
        "torch is installed).",
    )


def _ct2_model_from_env(env_var: str) -> str | None:
    """Return a usable CT2 model path from ``env_var`` or ``None``."""
    path = os.environ.get(env_var)
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not (p / "model.bin").exists():
        return None
    return str(p)


# --------------------------------------------------------------------------
# Path fixtures (session-scoped; skip when the resource is unavailable).
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def main_model_path() -> str:
    path = _ct2_model_from_env(MAIN_MODEL_ENV)
    if path is None:
        pytest.skip(
            f"set {MAIN_MODEL_ENV} to a converted CrisperWhisper CT2 model "
            "directory (containing model.bin + tokenizer.json + config.json)"
        )
    return path


@pytest.fixture(scope="session")
def draft_model_path() -> str:
    path = _ct2_model_from_env(DRAFT_MODEL_ENV)
    if path is None:
        pytest.skip(
            f"set {DRAFT_MODEL_ENV} to a converted CT2 draft model directory "
            "to run speculative-decoding tests"
        )
    return path


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    if _SAMPLES_DIR is None or not _SAMPLES_DIR.is_dir():
        pytest.skip("samples/ audio directory not found (set CW2_SAMPLES_DIR)")
    return _SAMPLES_DIR


@pytest.fixture(scope="session")
def en_audio(samples_dir: Path) -> str:
    """Path to a short (~18 s) English speech sample."""
    f = samples_dir / "example_en_1.wav"
    if not f.exists():
        pytest.skip(f"sample not found: {f}")
    return str(f)


# --------------------------------------------------------------------------
# Model fixtures (session-scoped; loading is expensive).
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model(main_model_path: str):
    """A ``CrisperWhisperModel`` backed by the main model only."""
    from crisperwhisper import CrisperWhisperModel

    return CrisperWhisperModel(main_model_path)


@pytest.fixture(scope="session")
def spec_model(main_model_path: str, draft_model_path: str):
    """A ``CrisperWhisperModel`` with a draft model for speculative decoding."""
    from crisperwhisper import CrisperWhisperModel

    return CrisperWhisperModel(
        main_model_path,
        draft_model=draft_model_path,
        speculative_k=5,
    )


# --------------------------------------------------------------------------
# Transformers backend fixtures.
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hf_model_path() -> str:
    """HuggingFace model id/dir for the transformers backend."""
    path = os.environ.get(HF_MODEL_ENV)
    if not path:
        pytest.skip(
            f"set {HF_MODEL_ENV} to a CrisperWhisper v2 HuggingFace model "
            "id or directory to run the transformers-backend tests"
        )
    return path


@pytest.fixture(scope="session")
def tf_engine(hf_model_path: str):
    """A low-level :class:`TransformersEngine` (skips without torch)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from crisperwhisper.transformers_engine import TransformersEngine

    return TransformersEngine(hf_model_path)


@pytest.fixture(scope="session")
def tf_model(hf_model_path: str):
    """A ``CrisperWhisperModel`` on the transformers backend."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from crisperwhisper import CrisperWhisperModel

    return CrisperWhisperModel(hf_model_path, backend="transformers")


# --------------------------------------------------------------------------
# Legacy (v1) backend fixtures.
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def v1_model_path() -> str:
    """HuggingFace id/dir for the legacy (v1) CrisperWhisper model."""
    path = os.environ.get(V1_MODEL_ENV)
    if not path:
        pytest.skip(
            f"set {V1_MODEL_ENV} to the legacy CrisperWhisper model "
            "(e.g. 'nyrahealth/CrisperWhisper') to run the v1 tests"
        )
    return path


@pytest.fixture(scope="session")
def v1_engine(v1_model_path: str):
    """A low-level :class:`TransformersEngine` on the legacy v1 model."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from crisperwhisper.transformers_engine import TransformersEngine

    return TransformersEngine(v1_model_path)


@pytest.fixture(scope="session")
def v1_model(v1_model_path: str):
    """A ``CrisperWhisperModel`` on the legacy (v1) model.

    Loading a v1 model emits a ``DeprecationWarning`` by design, so it is
    filtered here to keep the fixture setup quiet.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import warnings

    from crisperwhisper import CrisperWhisperModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return CrisperWhisperModel(v1_model_path)
