"""Automatic conversion from HuggingFace Whisper checkpoints to CTranslate2 format.

On first load the converter runs once and caches the result so subsequent
instantiations are near-instant.  If the user passes a directory that already
contains a CTranslate2 model (``model.bin``), conversion is skipped entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path(os.environ.get(
    "CRISPERWHISPER_CACHE",
    Path.home() / ".cache" / "crisperwhisper",
))

_CT2_SENTINEL = "model.bin"

_TOKENIZER_FILES = [
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "config.json",
]


def _is_ct2_model(path: Path) -> bool:
    return (path / _CT2_SENTINEL).exists()


def _cache_key(model_id: str, quantization: str) -> str:
    slug = model_id.replace("/", "--").replace("\\", "--")
    h = hashlib.sha256(model_id.encode()).hexdigest()[:12]
    return f"{slug}_{quantization}_{h}"


def _resolve_hf_or_local(model_name_or_path: str) -> Path:
    """Return a local directory for the model, downloading from HF Hub if needed."""
    p = Path(model_name_or_path)
    if p.is_dir():
        return p

    from huggingface_hub import snapshot_download

    logger.info("Downloading %s from HuggingFace Hub ...", model_name_or_path)
    local = snapshot_download(
        model_name_or_path,
        allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "*.model"],
    )
    return Path(local)


def _find_tokenizer_dir(model_dir: Path) -> Path:
    """CrisperWhisper checkpoints sometimes keep tokenizer files in the parent."""
    if (model_dir / "vocab.json").exists() or (model_dir / "tokenizer.json").exists():
        return model_dir
    parent = model_dir.parent
    if (parent / "vocab.json").exists() or (parent / "tokenizer.json").exists():
        return parent
    return model_dir


def _copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    for fname in _TOKENIZER_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def _sanitize_ct2_configs(ct2_dir: Path) -> None:
    """Remove JSON entries that crash CTranslate2's internal C++ JSON parser.

    CTranslate2 reads ``config.json`` and ``generation_config.json`` at
    generate-time but its nlohmann/json usage chokes on ``null`` values
    and JSON booleans (``true``/``false``) where it expects numbers.
    We strip any key whose value is ``None``, ``bool``, or a list
    containing ``None``.
    """
    for fname in ("config.json", "generation_config.json"):
        cfg_path = ct2_dir / fname
        if not cfg_path.exists():
            continue

        cfg = json.loads(cfg_path.read_text())
        keys_to_remove = []
        for k, v in cfg.items():
            if v is None or isinstance(v, bool):
                keys_to_remove.append(k)
            elif isinstance(v, list) and _list_has_null(v):
                keys_to_remove.append(k)

        if keys_to_remove:
            for k in keys_to_remove:
                del cfg[k]
            cfg_path.write_text(json.dumps(cfg, indent=2))
            logger.info(
                "Sanitized %s: removed %s",
                fname, ", ".join(keys_to_remove),
            )


def _list_has_null(lst: list) -> bool:
    for item in lst:
        if item is None:
            return True
        if isinstance(item, list) and _list_has_null(item):
            return True
    return False


def _ensure_tokenizer_json(ct2_dir: Path, source_dir: Path) -> None:
    """Build ``tokenizer.json`` (HuggingFace fast-tokenizer format) if missing.

    CrisperWhisper checkpoints ship with the slow tokenizer files
    (``vocab.json``, ``merges.txt``, ``added_tokens.json``, etc.) but
    CTranslate2 / the ``tokenizers`` library need ``tokenizer.json``.
    We use ``transformers.AutoTokenizer`` to load from the slow files and
    save the fast format.
    """
    if (ct2_dir / "tokenizer.json").exists():
        return

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning(
            "tokenizer.json missing and transformers not installed — "
            "cannot build fast tokenizer automatically."
        )
        return

    for search in (ct2_dir, source_dir):
        if (search / "vocab.json").exists() or (search / "tokenizer_config.json").exists():
            logger.info("Building tokenizer.json from slow tokenizer files in %s", search)
            tok = AutoTokenizer.from_pretrained(str(search))
            tok.save_pretrained(str(ct2_dir))
            return

    logger.warning("No tokenizer files found — tokenizer.json not created.")


def ensure_ct2_model(
    model_name_or_path: str,
    quantization: str = "float16",
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the path to a CTranslate2-format model, converting if necessary.

    Parameters
    ----------
    model_name_or_path
        HuggingFace model ID (e.g. ``nyrahealth/CrisperWhisper``) or local
        directory containing either HF-format or CT2-format weights.
    quantization
        One of ``float16``, ``float32``, ``int8``, ``int8_float16``,
        ``int8_float32``, ``int8_bfloat16``.
    cache_dir
        Where to store converted models.  Defaults to
        ``~/.cache/crisperwhisper/`` or ``$CRISPERWHISPER_CACHE``.
    """
    model_dir = _resolve_hf_or_local(model_name_or_path)

    if _is_ct2_model(model_dir):
        logger.info("Model at %s is already in CT2 format.", model_dir)
        return model_dir

    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _cache_key(model_name_or_path, quantization)
    ct2_dir = cache_root / key

    marker = ct2_dir / ".conversion_complete"
    if ct2_dir.exists() and marker.exists():
        logger.info("Using cached CT2 model at %s", ct2_dir)
        return ct2_dir

    logger.info(
        "Converting %s -> CT2 (%s) at %s ...",
        model_dir, quantization, ct2_dir,
    )

    try:
        import ctranslate2
    except ImportError as exc:
        raise ImportError(
            "ctranslate2 is required for model conversion. "
            "Install the CrisperWhisper fork with: "
            "pip install ctranslate2-crisperwhisper"
        ) from exc

    try:
        converter = ctranslate2.converters.TransformersConverter(
            str(model_dir),
            copy_files=None,
            load_as_float16=quantization in ("float16", "int8_float16", "int8_bfloat16"),
        )
        _orig_load_model = converter.load_model

        def _patched_load_model(model_class, name, **kwargs):
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return _orig_load_model(model_class, name, **kwargs)

        converter.load_model = _patched_load_model
    except Exception:
        try:
            from transformers import WhisperForConditionalGeneration
            _ = WhisperForConditionalGeneration  # verify import
        except ImportError as exc2:
            raise ImportError(
                "transformers and torch are required for first-time model conversion. "
                "Install with: pip install crisperwhisper[convert]"
            ) from exc2
        raise

    ct2_dir.mkdir(parents=True, exist_ok=True)
    converter.convert(str(ct2_dir), quantization=quantization, force=True)

    tokenizer_dir = _find_tokenizer_dir(model_dir)
    _copy_tokenizer_files(tokenizer_dir, ct2_dir)

    if tokenizer_dir != model_dir:
        _copy_tokenizer_files(model_dir, ct2_dir)

    _ensure_tokenizer_json(ct2_dir, model_dir)
    _sanitize_ct2_configs(ct2_dir)

    meta = {
        "source": model_name_or_path,
        "quantization": quantization,
        "source_dir": str(model_dir),
    }
    (ct2_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2))
    marker.touch()

    logger.info("Conversion complete: %s", ct2_dir)
    return ct2_dir
