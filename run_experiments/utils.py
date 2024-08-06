import copy
import json
import os
import string
from typing import Any, Dict, Generator, List, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from crisper_whisper import WhisperForConditionalGenerationWithAttentionLoss
from evaluate_word_segmentation import (
    TimestampedOutputs,
    convert_timestamps_from_labels_json_to_TimestampedOutput,
)
from loguru import logger
from transformers import (
    AutoConfig,
    AutomaticSpeechRecognitionPipeline,
    AutoProcessor,
    GenerationConfig,
    pipeline,
)


def get_mixing_scale(
    primary_utterance: Union[torch.Tensor, np.array],
    secondary_utterance: Union[torch.Tensor, np.array],
    snr: float,
) -> Any:
    energy_primary_utterance = float(torch.mean(torch.square(primary_utterance)))
    energy_secondary_utterance = float(torch.mean(torch.square(secondary_utterance)))
    mixing_scale = np.sqrt(
        energy_primary_utterance
        / (np.power(10, snr / 10) * energy_secondary_utterance + 1e-8)
    )
    mixing_scale = np.min(np.array([mixing_scale, 100]))
    return mixing_scale


def remove_punctuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).strip()


def find_matching_indexes(row: dict[str, Any]) -> list[int]:
    return [
        i
        for i, (pred, gt) in enumerate(zip(row["predictions"], row["gts"]))
        if remove_punctuation(pred[0].lower()) == remove_punctuation(gt[0].lower())
    ]


def common_indexes(group: Any) -> list[set[Any]]:
    # Use set intersection to find common elements in all lists in the group
    common_set = set(group.iloc[0])
    for idx_list in group.iloc[1:]:
        common_set.intersection_update(idx_list)
    return list(common_set)


def add_score_metrics(row: dict[str, Any], attribute: str) -> Any:
    return sum(
        getattr(row["metrics"][k], attribute)
        for k in row["matching_indexes_common"]
        if k < len(row["metrics"])
    )


def add_gaussian_noise_to_audio(primary_utterance_path: str, snr: float) -> np.ndarray:
    """
    Adds Gaussian noise to an audio signal at a specified Signal-to-Noise Ratio (SNR).

    Args:
        primary_utterance_path (str): Path to the primary utterance audio file.
        snr (float): Desired Signal-to-Noise Ratio (SNR) in decibels.

    Returns:
        np.ndarray: The noisy audio signal.
    """
    wave, sr = torchaudio.load(primary_utterance_path)
    primary_utterance = copy.deepcopy(wave.flatten())
    noise = torch.tensor(np.random.normal(0, 1, len(primary_utterance)))
    mixing_scale = get_mixing_scale(primary_utterance, noise, snr)
    primary_utterance += noise * mixing_scale
    return np.array(primary_utterance)


def get_alignment_head_generator(
    num_layers: int, num_heads: int
) -> Generator[List[List[int]], None, None]:
    """
    Generates individual alignment head configurations for a Whisper Model.

    Args:
        num_layers (int): The number of transformer layers in the model.
        num_heads (int): The number of heads in each layer.

    Yields:
        Generator[List[List[int]], None, None]: A generator that yields lists containing pairs of layer and head indices.
    """
    for num_layer in range(num_layers):
        for num_head in range(num_heads):
            yield [[num_layer, num_head]]


def load_labels(
    dataset_path: str, dataset: str, split: str = "test", limit: int = 100
) -> list[dict[str, Any]]:
    labels_path = os.path.join(dataset_path, dataset, "labels.json")
    with open(labels_path, "r") as f:
        labels = json.load(f)
    logger.info(f"Total number of labels in {labels_path}: {len(labels)}")
    filtered_labels = [
        label for label in labels if label["split"] == split and "timings" in label
    ][:limit]
    logger.info(
        f"Number of loaded labels (split={split}, limit={limit}): {len(filtered_labels)}."
    )
    return filtered_labels


def prepare_ground_truth(labels: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    ground_truth = []
    for label in labels:
        gt_transcript = label["transcript"]
        gt_timestamps = label.get("timings", [])
        gt_timestamps = convert_timestamps_from_labels_json_to_TimestampedOutput(
            gt_timestamps
        )
        ground_truth.append((gt_transcript, gt_timestamps))
    return ground_truth


def configure_model(
    model_path: str, device: str
) -> WhisperForConditionalGenerationWithAttentionLoss:
    model: WhisperForConditionalGenerationWithAttentionLoss = (
        WhisperForConditionalGenerationWithAttentionLoss.from_pretrained(
            model_path, ignore_mismatched_sizes=False
        )
    )
    model.to(device)
    model.eval()
    return model


def get_processor_config_and_genconfig(
    model_path: str,
) -> tuple[AutoProcessor, AutoConfig, GenerationConfig]:
    processor = AutoProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    return processor, config, generation_config


def setup_pipeline(
    model: WhisperForConditionalGenerationWithAttentionLoss,
    processor: AutoProcessor,
    device: str,
    chunk_length_s: int = 30,
    batch_size: int = 60,
) -> AutomaticSpeechRecognitionPipeline:
    dtype = torch.float32

    if "cuda" in device:
        dtype = torch.float16
        model.half()

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        device=device,
        torch_dtype=dtype,
    )


def process_audio_files(
    audio_paths: list[str], pipe: AutomaticSpeechRecognitionPipeline
) -> tuple[list[dict[str, Any]], list[int]]:
    result = []
    success_ids = []
    for id, audio_path in enumerate(audio_paths):
        try:
            result.append(pipe(audio_path, return_timestamps="word"))
            logger.info(f"Processed audio file {audio_path}")
            success_ids.append(id)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    return result, success_ids


def adjust_pauses(
    predicted_transcripts_and_timestamps: list[tuple[str, TimestampedOutputs]],
    threshold: float = 0.08,
) -> list[Any]:
    new_predictions = []
    for element in predicted_transcripts_and_timestamps:
        new_element = copy.deepcopy(element)
        new_element[1].adjust_pauses(split_threshold=threshold)
        new_predictions.append(new_element)
    return new_predictions


def configure_model_generation(
    model: WhisperForConditionalGenerationWithAttentionLoss,
    model_name: str,
    processor: AutoProcessor,
    alignment_heads: list[Any],
    median_filter_width: int,
) -> None:
    """
    Configures the generation settings of the model based on its name.

    Args:
        model: The model to configure.
        model_name: The name of the model.
        processor: the WhisperProcessor
    """
    if "crisper" not in model_name:
        model.generation_config.legacy = False
        tokens_to_ignore_for_dtw = ["°"]
    else:
        model.generation_config.legacy = True
        tokens_to_ignore_for_dtw = ["?", "!", ",", "."]

    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )

    model.generation_config.token_ids_to_ignore_for_dtw = (
        processor.tokenizer.convert_tokens_to_ids(tokens_to_ignore_for_dtw)
    )
    model.generation_config.alignment_heads = alignment_heads
    model.generation_config.median_filter_width = median_filter_width


def top_heads_by_f1(
    df: pd.DataFrame, num_heads: int = 20, collar: float = 0.1
) -> Dict[str, List[int]]:
    """
    Get the top `num_heads` alignment heads for each model based on F1 scores.

    Args:
        df (pd.DataFrame): DataFrame containing the model results with columns 'model', 'F1_collar.2', and 'heads'.
        num_heads (int): The number of top heads to retrieve for each model. Defaults to 20.

    Returns:
        Dict[str, List[int]]: Dictionary where keys are model names and values are lists of top alignment head indices.
    """
    results: Dict[str, List[int]] = {}

    # Group by 'model', then sort each group by 'F1_collar.2' and take the top `num_heads` rows
    grouped = df.groupby("model")
    for model, group in grouped:
        top_heads = group.sort_values(f"F1_collar.{collar}", ascending=False).head(
            num_heads
        )
        # Extract only the 'heads' column and convert it to a list of integers
        results[model] = [head[0] for head in top_heads["heads"]]

    return results
