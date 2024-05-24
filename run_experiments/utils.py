import copy
import json
import os
import re
import string
import sys
from typing import Callable, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
import tqdm
from num2words import num2words
from transformers import AutoConfig, AutoProcessor, GenerationConfig, pipeline

from crisper_whisper import WhisperForConditionalGenerationWithAttentionLoss
from evaluate_word_segmentation import convert_timestamps_from_labels_json_to_TimestampedOutput





def get_mixing_scale(primary_utterance: Union[torch.Tensor, np.array],
                     secondary_utterance: Union[torch.Tensor, np.array],
                     snr: int):
    energy_primary_utterance = float(torch.mean(torch.square(primary_utterance)))
    energy_secondary_utterance = float(torch.mean(torch.square(secondary_utterance)))
    mixing_scale = np.sqrt(energy_primary_utterance / (np.power(10, snr / 10) * energy_secondary_utterance + 1e-8))
    mixing_scale = np.min(np.array([mixing_scale,100]))
    return mixing_scale

    
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).strip()

def find_matching_indexes(row):
    return [i for i, (pred, gt) in enumerate(zip(row['predictions'], row['gts'])) if remove_punctuation(pred[0].lower()) == remove_punctuation(gt[0].lower())]

def common_indexes(group):
    # Use set intersection to find common elements in all lists in the group
    common_set = set(group.iloc[0])
    for idx_list in group.iloc[1:]:
        common_set.intersection_update(idx_list)
    return list(common_set)


def add_score_metrics(row, attribute):
    return sum(getattr(row['metrics'][k], attribute) for k in row['matching_indexes_common'] if k < len(row['metrics']))


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
    
def get_alignment_head_generator(num_layers: int, num_heads: int) -> Generator[List[List[int]], None, None]:
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

def load_labels(dataset_path: str, dataset: str, split: str = 'test', limit: int = 100):
    labels_path = os.path.join(dataset_path, dataset, 'labels.json')
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return [label for label in labels if label['split'] == split and 'labels' in label][:limit]

def prepare_ground_truth(labels: list):
    ground_truth = []
    for label in labels:
        gt_transcript = label['transcript']
        gt_timestamps = label.get("labels", [])
        gt_timestamps = convert_timestamps_from_labels_json_to_TimestampedOutput(gt_timestamps)
        ground_truth.append((gt_transcript, gt_timestamps))
    return ground_truth

def configure_model(model_path: str, device: str):
    model = WhisperForConditionalGenerationWithAttentionLoss.from_pretrained(model_path, ignore_mismatched_sizes=False)
    model.to(device)
    model.eval()
    return model

def get_processor_config_genConfig(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    return processor, config, generation_config

def setup_pipeline(model, processor, device, chunk_length_s=30, batch_size=60):
    return pipeline(
        "automatic-speech-recognition",
        model=model.half(),
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        device=device,
        torch_dtype=torch.float16
    )

def process_audio_files(audio_paths, pipe):
    try:
        return pipe(audio_paths, return_timestamps="word")
    except Exception as e:
        print(f"Error processing audio: {e}")
        return []

def adjust_pauses(predicted_transcripts_and_timestamps, threshold = 0.08):
    new_predictions = []
    for element in predicted_transcripts_and_timestamps:
        new_element = copy.deepcopy(element)
        new_element[1].adjust_pauses(split_threshold=threshold)
        new_predictions.append(new_element)
    return new_predictions

def configure_model_generation(model, model_name, processor, alignment_heads, median_filter_width):
    """
    Configures the generation settings of the model based on its name.
    
    Args:
        model: The model to configure.
        model_name: The name of the model.
        processor: the WhisperProcessor
    """
    if 'crisper' not in model_name:
        model.generation_config.legacy = True
        tokens_to_ignore_for_dtw = ["°"]
    else:
        model.generation_config.legacy = False
        tokens_to_ignore_for_dtw = ["?", "!", ",", "."]
    
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    
    model.generation_config.token_ids_to_ignore_for_dtw = processor.tokenizer.convert_tokens_to_ids(
        tokens_to_ignore_for_dtw
    )
    model.generation_config.alignment_heads = alignment_heads
    model.generation_config.median_filter_width = median_filter_width

def top_heads_by_f1(df: pd.DataFrame, num_heads: int = 20, collar=.1) -> Dict[str, List[int]]:
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
    grouped = df.groupby('model')
    for model, group in grouped:
        top_heads = group.sort_values(f'F1_collar.{collar}', ascending=False).head(num_heads)
        # Extract only the 'heads' column and convert it to a list of integers
        results[model] = [head[0] for head in top_heads['heads']]

    return results


