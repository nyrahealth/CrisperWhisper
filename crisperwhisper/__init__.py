"""CrisperWhisper — fast inference for CrisperWhisper speech recognition models."""

from crisperwhisper import _nvidia_libs

# Must run before anything imports ctranslate2 (see _nvidia_libs docstring),
# hence the imports below the call.
_nvidia_libs.preload()

from crisperwhisper.model import (  # noqa: E402
    DEFAULT_MODEL,
    OFFICIAL_MODELS,
    CrisperWhisperModel,
)
from crisperwhisper.result import (  # noqa: E402
    TranscriptionResult,
    ChunkResult,
    WordTimestamp,
)

__all__ = [
    "CrisperWhisperModel",
    "TranscriptionResult",
    "ChunkResult",
    "WordTimestamp",
    "DEFAULT_MODEL",
    "OFFICIAL_MODELS",
]
__version__ = "2.0.1"


def check_speculative_support() -> None:
    """Verify the CrisperWhisper CT2 fork is installed for speculative decoding.

    Raises ``ImportError`` if the installed ctranslate2 lacks the custom
    speculative-decoding APIs (prefill, forward_step_greedy, etc.).
    Called lazily when speculative decoding is first requested.
    """
    import ctranslate2

    whisper_cls = getattr(ctranslate2.models, "Whisper", None)
    if whisper_cls is None:
        raise ImportError(
            "ctranslate2 Whisper model not available — "
            "install ctranslate2-crisperwhisper for speculative decoding."
        )

    required = ("prefill", "forward_step_greedy", "forward_batch_greedy")
    missing = [m for m in required if not hasattr(whisper_cls, m)]
    if missing:
        raise ImportError(
            f"The installed ctranslate2 ({ctranslate2.__version__}) is missing "
            f"speculative-decoding APIs: {', '.join(missing)}.\n"
            "You likely have the upstream package installed on top of the fork "
            "(both own the ctranslate2 import directory, so the last install "
            "wins). Fix with:\n"
            "  pip uninstall -y ctranslate2\n"
            "  pip install --force-reinstall ctranslate2-crisperwhisper"
        )
