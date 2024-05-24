
import os
import whisperx
import whisper_timestamped as whisper_timestamped

from abc import ABC, abstractmethod
from evaluate_word_segmentation import TimestampedOutput, TimestampedOutputs
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Union



class ModelOutput:
    """Represents the output of a speech recognition model."""
    def __init__(self, 
                 prediction_str: str, 
                 timestamped_outputs: TimestampedOutputs, 
                 raw_output: Dict[str, Any]):
        self.prediction_str = prediction_str
        self.timestamped_outputs = timestamped_outputs
        self.raw_output = raw_output

    def to_json(self) -> Dict[str, Any]:
        """Converts the model output to a JSON-serializable dictionary."""
        return {
            "prediction_str": self.prediction_str,
            "timestamped_outputs": self.timestamped_outputs.to_json(),
            "raw_output": self.raw_output
        }


class NormalizedOutput:
    """Represents normalized output with timestamps."""
    def __init__(self, 
                 prediction_str: str,
                 normalized_str: str, 
                 timestamped_outputs: TimestampedOutputs, 
                 normalized_timestamped_outputs: TimestampedOutputs, 
                 raw_output: Dict[str, Any],
                 sample_id: str):
        self.prediction_str = prediction_str
        self.normalized_str = normalized_str
        self.timestamped_outputs = timestamped_outputs
        self.normalized_timestamped_outputs = normalized_timestamped_outputs
        self.raw_output = raw_output
        self.sample_id = sample_id

    def to_json(self) -> Dict[str, Any]:
        """Converts the model output to a JSON-serializable dictionary."""
        return {
            "prediction_str": self.prediction_str,
            "normalized_str": self.normalized_str,
            "normalized_timestamped_outputs": self.normalized_timestamped_outputs.to_json(),
            "timestamped_outputs": self.timestamped_outputs.to_json(),
            "raw_output": self.raw_output,
            "sample_id": self.sample_id
        }



class ModelConfig:
    """Defines the configuration for speech recognition models."""
    WHISPER_X = {
       # "type": "whisper_x",
        "model_name_whisper": "large-v2",
        "model_name_align": "WAV2VEC2_ASR_BASE_960H",
        "language_code": "en"
    }
    
    CRISPER_WHISPER = {
        #"type": "crisper_whisper",
        "checkpoint_path": "/home/azureuser/laurin/code/research/output/crisper_whisper_timestamp_finetuned",
        "language_code": "en"
    }
    
    TIMESTAMPED_WHISPER = {
        #"type": "crisper_whisper",
        "size": "openai/whisper-large-v2",
        "language_code": "en"
    }


class SpeechRecognitionModel(ABC):
    """Abstract base class for speech recognition models."""
    def __init__(self, name: str, language_code: str = "en"):
        self.name = name
        self.language_code = language_code

    @abstractmethod
    def __call__(self, audio_path: str, batch_size: int = 1) -> ModelOutput:
        """Method to process audio file and return the model output."""
        pass


class WhisperXModel(SpeechRecognitionModel):
    """Speech recognition model based on WhisperX."""
    def __init__(self, model_name_whisper: str, model_name_align: str, 
                 language_code: str = "en", device: str = "cuda", 
                 compute_type: str = "float16"):
        model_name = f"whisper_x_{model_name_whisper}_{model_name_align}"
        super().__init__(name=model_name, language_code=language_code)
        self.model_name_whisper = model_name_whisper
        self.model_name_align = model_name_align
        self.model = whisperx.load_model(
            model_name_whisper, device, compute_type=compute_type, language="en")
        self.device = device
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code="en", device=device, model_name=model_name_align)

    def _format_output(self, transcript: Dict[str, Any]) -> ModelOutput:
        """Formats the raw model transcript into a structured ModelOutput."""
        timestamped_outputs = [TimestampedOutput(
            word=entry["word"], 
            start=entry["start"], 
            end=entry["end"]) for entry in transcript["word_segments"]]
        
        prediction_str = " ".join([entry["word"] for entry in transcript["word_segments"]])
        return ModelOutput(
            prediction_str=prediction_str, 
            timestamped_outputs=TimestampedOutputs(timestamped_outputs), 
            raw_output=transcript)

    def __call__(self, audio_path: Union[str, Path, List[float]], batch_size: int = 1) -> ModelOutput:
        """Processes an audio file or raw audio data and returns the transcribed output."""
        if isinstance(audio_path, (str, Path)):
            audio = whisperx.load_audio(audio_path)
        else:
            audio = audio_path

        transcription_result = self.model.transcribe(audio, batch_size=batch_size)
        aligned_result = whisperx.align(
            transcription_result["segments"], 
            self.model_a, 
            self.metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        return self._format_output(aligned_result)

class WhisperTimestamped(SpeechRecognitionModel):
    """Whisper Timestamped model"""
    def __init__(self, size: str, 
                 language_code: str = "en", device: str = "cuda"):
        model_name = f"whisper_timestamped_{size}"
        super().__init__(name=model_name, language_code=language_code)
        self.model = whisper_timestamped.load_model(size, device=device)
        self.language_code = language_code
        
    def __call__(self, audio_path: Union[str, Path, List[float]], batch_size: int = 1) -> ModelOutput:
        """Processes an audio file or raw audio data and returns the transcribed output."""

        
    def __call__(self, audio_path: str, batch_size: int=1) -> ModelOutput:
        if isinstance(audio_path, (str, Path)):
            audio = whisper_timestamped.load_audio(audio_path)
        else:
            audio = audio_path
        result = whisper_timestamped.transcribe(self.model,
                          audio,
                          naive_approach=True,
                          language=self.language_code,
                          beam_size=1, 
                          refine_whisper_precision=0) 
        
        return self._format_output(result)
        
        
    def _format_output(self, transcript: Dict[str, Any]) -> ModelOutput:
        """Formats the raw model transcript into a structured ModelOutput."""
        timestamped_outputs = []
        for segment in transcript["segments"]:
            for entry in segment["words"]:
                tso = TimestampedOutput(
                    word=entry["text"], start=entry["start"], end=entry["end"])
                timestamped_outputs.append(tso)
        
        model_output = ModelOutput(
            prediction_str=transcript["text"], 
            timestamped_outputs=TimestampedOutputs(timestamped_outputs), 
            raw_output=transcript)
        
        return model_output
        
        
def get_model(model_config: ModelConfig, 
              device: str = "cuda") -> SpeechRecognitionModel:
    """Factory method to create a speech recognition model."""
    if model_config["type"] == "whisper_x":
        return WhisperXModel(
            model_name_whisper=model_config["model_name_whisper"], 
            model_name_align=model_config["model_name_align"], 
            language_code=model_config["language_code"], device=device)
    elif model_config["type"] == "whisper_timestamped":
        return  WhisperTimestamped(model_config['size'],
                                   language_code=model_config['language_code'], 
                                   device = device)

    
    raise ValueError(f"Unsupported model type: {model_config['type']}")


def transcribe_single_file(model: SpeechRecognitionModel, audio_path: str) -> ModelOutput:
    """Transcribes a single audio file and saves the output to a specified location."""
    model_output = model(audio_path=audio_path)
    return model_output


def transcribe_speech_files(
        model: SpeechRecognitionModel, dataset_name: str, audio_paths: List[str]) -> List[str]:
    """Transcribes a list of audio files and saves the outputs in a structured directory."""
    save_to = os.path.join("predictions", dataset_name, model.name, "transcripts")
    os.makedirs(save_to, exist_ok=True)
    error_ids = []
    outputs = []
    for audio_path in tqdm(audio_paths, desc="Transcribing audio files"):
        try:
            outputs.append(transcribe_single_file(
                model=model, audio_path=audio_path))
        except KeyError:
            error_ids.append(audio_path)
    return outputs, error_ids

