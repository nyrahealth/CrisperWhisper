import copy
import json
import re
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import attrs
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from num2words import num2words

# Path to the 'src' folder inside the 'transformers' submodule
transformers_path = Path(__file__).parent.parent / "transformers/src"

# Add this path to the front of sys.path
sys.path.insert(0, str(transformers_path))


class AllowedCharacters(Enum):
    GERMAN_CASED: str = "abcdefghijklmnopqrstuvwxyzäöüABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"
    GERMAN_UNCASED: str = "abcdefghijklmnopqrstuvwxyzäöü"
    ENGLISH_CASED: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ENGLISH_UNCASED: str = "abcdefghijklmnopqrstuvwxyz"


class SpecialCharacters(Enum):
    PUNCTUATION: str = "[,?.!;:\*()~\"\-',]"
    GERMAN_CASED: str = (
        "[^a-zA-ZäöüÄÖÜß\\s]"  # all characters except a,...,z,A,...,Z,ä,ö,ü,ß,Ä,Ö,Ü
    )
    GERMAN_UNCASED: str = "[^a-zäöüß\\s]"  # all characters except a,...,z,ä,ö,ü,ß
    ENGLISH_CASED: str = "[^a-zA-Z\\s]"  # all characters except a,...,z,A,...,Z
    ENGLISH_UNCASED: str = "[^a-z\\s]"  # all characters except a,...,z


class TextPreprocessor:
    """Class for preprocessing text files in a repeatable manner.

    Attributes:
        preprocessing_methods (Optional[list[str]]): A list of strings, specifying pre-processing methods. The
            order of these methods is important as they are executed sequentially.
    """

    def __init__(
        self,
        preprocessing_methods: Optional[list[str]] = None,
        tokens_not_to_be_preprocessed: list[str] = [],
        tokens_mapping: dict[str, Any] = {},
        language: str = "en",
    ) -> None:
        self.preprocessing_methods = preprocessing_methods
        self.tokens_not_to_be_preprocessed = tokens_not_to_be_preprocessed
        self.tokens_mapping = tokens_mapping
        self.language = language

    def __call__(self, text: str) -> str:
        return self.preprocess(text)

    def replace_nums_in_text(self, text: str) -> str:
        numbers = re.findall(r"[0-9]+", text)
        for number in numbers:
            line = re.sub(number, num2words(number, lang=self.language), text)
            text = line
        return text

    @staticmethod
    def token_to_lowercase_a_to_z_placeholder(token: str) -> str:
        timestamp = int(time.time())
        combined = f"{token}_{timestamp}"

        combined_sum = sum(ord(char) for char in combined)
        placeholder = ""
        while combined_sum > 0:
            combined_sum, remainder = divmod(combined_sum, 26)
            placeholder += chr(97 + remainder)
        return placeholder * 5

    @staticmethod
    def to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def replace_special_characters(
        text: str, chars_to_replace: Optional[dict[str, str]] = None
    ) -> str:
        """Maps characters from keys to values."""
        if chars_to_replace is None:
            chars_to_replace = {"ß": "ss"}
        for to_replace, replace_with in chars_to_replace.items():
            text = re.sub(to_replace, replace_with, text)
        return text

    @staticmethod
    def phonemization_fixes(text: str, fixes: list[Callable]) -> str:  # type: ignore
        for fix in fixes:
            text = fix(text)
        return text

    @staticmethod
    def replace_c_with_ze(sent: str = "") -> str:
        patterns = [re.compile(r" c "), re.compile(r" c$"), re.compile(r"^c ")]
        output = re.sub(patterns[0], " ze ", sent)
        output = re.sub(patterns[1], " ze", output)
        output = re.sub(patterns[2], "ze ", output)
        return output

    @staticmethod
    def replace_pc_with_peze(sent: str = "") -> str:
        patterns = [re.compile(r" pc "), re.compile(r"( pc)$"), re.compile(r"^(pc )")]
        output = re.sub(patterns[0], " peze ", sent)
        output = re.sub(patterns[1], " peze", output)
        output = re.sub(patterns[2], "peze ", output)
        return output

    @staticmethod
    def remove_unnecessary_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_all_spaces(text: str) -> str:
        return text.replace(" ", "")

    @staticmethod
    def remove_special_characters(
        text: str,
        characters_to_ignore: SpecialCharacters = SpecialCharacters.PUNCTUATION,
    ) -> str:
        """Removes all characters `characters_to_ignore`."""
        return re.sub(characters_to_ignore.value, "", text)

    def apply_text_processing_method(self, preprocessing_method: str, text: str) -> str:
        if preprocessing_method == "lower":
            return self.to_lower(text)
        elif preprocessing_method == "remove_punctuation":
            return self.remove_special_characters(
                text=text, characters_to_ignore=SpecialCharacters.PUNCTUATION
            )
        elif preprocessing_method == "replace_special_chars":
            return self.replace_special_characters(text=text)
        elif preprocessing_method == "remove_non_german_chars":
            return self.remove_special_characters(
                text=text, characters_to_ignore=SpecialCharacters.GERMAN_UNCASED
            )
        elif preprocessing_method == "remove_non_english_chars":
            return self.remove_special_characters(
                text=text, characters_to_ignore=SpecialCharacters.ENGLISH_UNCASED
            )
        elif preprocessing_method == "replace_digits_with_words":
            return self.replace_nums_in_text(text=text)
        elif preprocessing_method == "phonemization_fixes":
            return self.phonemization_fixes(
                text=text, fixes=[self.replace_c_with_ze, self.replace_pc_with_peze]
            )
        elif preprocessing_method == "remove_unnecessary_spaces":
            return self.remove_unnecessary_spaces(text)
        elif preprocessing_method == "remove_all_spaces":
            return self.remove_all_spaces(text)
        else:
            raise NotImplementedError(
                f"{preprocessing_method} is not a valid pre-processing method."
            )

    def preprocess(self, text: str) -> str:
        """Applies the `preprocessing_methods` sequentially to `text`."""
        token_to_placeholder = {
            token: self.token_to_lowercase_a_to_z_placeholder(token)
            for token in self.tokens_not_to_be_preprocessed
        }
        text = str(text)
        for token, placeholder in token_to_placeholder.items():
            text = text.replace(token, placeholder)
        for preprocessing_method in self.preprocessing_methods:  # type: ignore
            text = self.apply_text_processing_method(preprocessing_method, text)
        for token, placeholder in token_to_placeholder.items():
            text = text.replace(placeholder, token)
        for token, mapping_token in self.tokens_mapping.items():
            text = text.replace(token, mapping_token)
        return str(text)


DEFAULT_NORMALIZER = TextPreprocessor(
    preprocessing_methods=[
        "lower",
        "remove_punctuation",
        "replace_special_chars",
        "remove_non_english_chars",
        "remove_unnecessary_spaces",
    ],
    tokens_not_to_be_preprocessed=["[UH]", "[UM]"],
    language="en",
)


@attrs.define
class EvaluationConfig:
    collar: float = attrs.field(default=0.2)
    transcripts_must_match: bool = attrs.field(default=True)
    do_normalize: bool = attrs.field(default=True)
    text_normalizer: Callable = attrs.field(default=DEFAULT_NORMALIZER)  # type: ignore
    max_number_of_evals: int = attrs.field(default=-1)
    pause_threshold: float = -1.0
    pause_type: str = "implicit"


class TimestampedOutput:
    """Represents a word with timestamps and an optional type."""

    def __init__(self, word: str, start: float, end: float, word_type: str | None = ""):
        self.word = word
        self.start = start
        self.end = end
        self.type = word_type

    def to_dict(self) -> dict[str, Any]:
        """Converts the timestamped output to a dictionary."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, result: dict[str, Any]) -> "TimestampedOutput":
        return cls(
            word=result["word"],
            start=result["start"],
            end=result["end"],
            word_type=result.get("type", ""),
        )

    def __str__(self) -> str:
        """String representation of the TimestampedOutput."""
        return f'Word: "{self.word}", Start: {self.start}, End: {self.end}, Type: {self.type or "N/A"}'


class TimestampedOutputs:
    """Collection of timestamped word outputs."""

    def __init__(self, entries: list[TimestampedOutput]):
        self.entries = entries

    def to_json(self) -> list[dict[str, Any]]:
        """Converts timestamped outputs to a JSON-serializable list."""
        return [entry.to_dict() for entry in self.entries]

    @classmethod
    def from_dict(cls, result: list[dict[str, Any]]) -> "TimestampedOutputs":
        entries = []
        for entry in result:
            tso = TimestampedOutput.from_dict(entry)
            entries.append(tso)
        return cls(entries=entries)

    def __str__(self) -> str:
        """String representation of the TimestampedOutputs."""
        return "\n".join(str(entry) for entry in self.entries)

    def clean_timestamped_outputs(
        self,
        text_normalizer: Callable = DEFAULT_NORMALIZER,  # type: ignore
    ) -> "TimestampedOutputs":
        result = []
        entries = copy.deepcopy(self.entries)
        for entry in entries:
            normalized_word = text_normalizer(entry.word)
            entry.word = normalized_word
            if normalized_word and len(normalized_word):
                result.append(entry)
        return TimestampedOutputs(result)

    def adjust_pauses(
        self, split_threshold: float = 0.06, pause_type: str = "implicit"
    ) -> None:
        """Adjust the timing of pauses (either actual spaces or implicit pauses) and exceed the threshold."""
        if split_threshold < 0:
            logger.info(
                f"Split threshold is negative ({split_threshold}), no pause adjustment was done."
            )
            return
        if pause_type == "explicit":
            for i in range(1, len(self.entries) - 1):
                entry = self.entries[i]
                if entry.word == " ":
                    duration = entry.end - entry.start
                    if duration > split_threshold:
                        distribute = split_threshold / 2
                    else:
                        distribute = duration / 2
                    # Check the previous and next words
                    if self.entries[i - 1].word != " ":
                        self.entries[i - 1].end += distribute
                    if self.entries[i + 1].word != " ":
                        self.entries[i + 1].start -= distribute

                    # Update the current pause's timing
                    entry.start += distribute
                    entry.end -= distribute
        elif pause_type == "implicit":
            for current_entry, next_entry in zip(self.entries[:-1], self.entries[1:]):
                duration = next_entry.start - current_entry.end
                if duration < 0:
                    raise ValueError(
                        f"Timestamps overlap: Start of next entry {next_entry.start} is later than end of current entry {next_entry.start}"
                    )

                if duration > split_threshold:
                    distribute = split_threshold / 2
                else:
                    distribute = duration / 2
                next_entry.start -= distribute
                current_entry.end += distribute
        else:
            raise ValueError(
                f"Pause type {pause_type} is not defined, must be 'implicit' or 'explicit'."
            )


def convert_timestamps_from_transformers_pipe_to_TimestampedOutput(
    timestamps: list[dict[str, Any]]
) -> TimestampedOutputs:
    converted_list = [
        {
            "word": timestamp["text"],
            "start": timestamp["timestamp"][0],
            "end": timestamp["timestamp"][1],
        }
        for timestamp in timestamps
        if timestamp.get("text") and timestamp.get("timestamp")
    ]
    updated_list = []
    for word1, word2 in zip(converted_list, converted_list[1:]):
        updated_list.append(word1)
        updated_list.append({"word": " ", "start": word1["end"], "end": word2["start"]})
    if len(converted_list) >= 1:
        updated_list.append(converted_list[-1])

    return TimestampedOutputs.from_dict(updated_list)


def convert_timestamps_from_labels_json_to_TimestampedOutput(
    timestamps: list[dict[str, Any]]
) -> TimestampedOutputs:
    converted_list = [
        {
            "word": timestamp["word"],
            "start": timestamp["starttime"],
            "end": timestamp["endtime"],
        }
        for timestamp in timestamps
        if timestamp.get("word")
        and timestamp.get("starttime", None) is not None
        and timestamp.get("endtime", None) is not None
    ]
    return TimestampedOutputs.from_dict(converted_list)


def do_overlap(
    start1: float, end1: float, start2: float, end2: float, collar: float = 0.2
) -> bool:
    if any([isinstance(val_, str) for val_ in [start1, start2, end1, end2, collar]]):
        raise TypeError("All inputs must be float or int, not str.")
    if start1 > end1 or start2 > end2:
        raise ValueError(
            "Invalid interval: start time must be less than or equal to end time"
        )
    if collar < 0:
        raise ValueError("Collar must be non-negative")
    overlap = (
        round(abs(start1 - start2), 4) <= collar
        and round(abs(end1 - end2), 4) <= collar
    )
    return overlap


def calculate_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    # Calculate the intersection of the two intervals
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    # Calculate the union of the two intervals
    total_start = min(start1, start2)
    total_end = max(end1, end2)
    union = total_end - total_start

    # Calculate and return the IoU
    iou = intersection / union if union != 0 else 0
    return iou


def get_precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall


def round_and_pad(number: float, decimals: int) -> str:
    rounded_number = round(number, decimals)
    formatted_number = f"{rounded_number:.{decimals}f}"
    return formatted_number


class PrecisionRecallMetrics:
    def __init__(
        self,
        number_of_instances: int,
        tp: int,
        fp: int,
        fn: int,
        ious: list[float] = [],
    ) -> None:
        self.number_of_instances = number_of_instances
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.ious = ious

    @property
    def precision(self) -> float:
        return self.get_precision_recall()[0]

    @property
    def recall(self) -> float:
        return self.get_precision_recall()[1]

    @property
    def f1_score(self) -> float:
        return self.calculate_f1_score()

    @property
    def avg_iou(self) -> float:
        return sum(self.ious) / len(self.ious) if len(self.ious) else 0.0

    def __add__(self, other: "PrecisionRecallMetrics") -> "PrecisionRecallMetrics":
        new_number_of_instances = self.number_of_instances + other.number_of_instances
        new_tp = self.tp + other.tp
        new_fp = self.fp + other.fp
        new_fn = self.fn + other.fn

        new_ious = self.ious + other.ious
        return PrecisionRecallMetrics(
            new_number_of_instances, new_tp, new_fp, new_fn, new_ious
        )

    def __radd__(self, other: "PrecisionRecallMetrics") -> "PrecisionRecallMetrics":
        if other == 0:  # type: ignore
            return self
        else:
            return self.__add__(other)

    def __str__(self) -> str:
        return (
            f"{self.number_of_instances}, {self.tp}, {self.fp}, {self.fn}, {round_and_pad(self.precision,3)},"
            f" {round_and_pad(self.recall,3)}, {round_and_pad(self.f1_score,3)}, {round_and_pad(self.avg_iou,3)}"
        )

    def get_precision_recall(self) -> tuple[float, float]:
        tp = self.tp
        fp = self.fp
        fn = self.fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        return precision, recall

    def calculate_f1_score(self) -> float:
        precision = self.precision
        recall = self.recall
        # Ensure the denominator is not zero to avoid division by zero error
        if precision + recall == 0:
            return 0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


def evaluate_segmentation(
    references: TimestampedOutputs,
    predictions: TimestampedOutputs,
    eval_config: EvaluationConfig,
) -> PrecisionRecallMetrics:
    collar = eval_config.collar
    if eval_config.do_normalize:
        references = references.clean_timestamped_outputs(eval_config.text_normalizer)
        predictions = predictions.clean_timestamped_outputs(eval_config.text_normalizer)
    if eval_config.pause_threshold > 0:
        predictions.adjust_pauses(
            split_threshold=eval_config.pause_threshold,
            pause_type=eval_config.pause_type,
        )
    tp = 0
    fn = 0
    ious = []

    # Marking predictions as matched to handle False Positives later
    prediction_matched = [False] * len(predictions.entries)

    for gt in references.entries:
        gt_word = gt.word  # Lowercase for case-insensitive comparison
        gt_start, gt_end = gt.start, gt.end
        found_match = False

        for i, pred in enumerate(predictions.entries):
            pred_word = pred.word  # Lowercase for case-insensitive comparison
            pred_start, pred_end = pred.start, pred.end

            if (
                gt_word == pred_word and not prediction_matched[i]
            ):  # Ensure each prediction is only counted once
                if do_overlap(gt_start, gt_end, pred_start, pred_end, collar=collar):
                    tp += 1
                    prediction_matched[i] = True
                    found_match = True
                    ious.append(calculate_iou(gt_start, gt_end, pred_start, pred_end))

                    break  # Move to the next ground truth word after finding a match
                else:
                    ious.append(0.0)
        if not found_match:
            fn += 1

    # Count False Positives
    fp = prediction_matched.count(False)

    pr_metrics = PrecisionRecallMetrics(
        number_of_instances=len(references.entries),
        tp=tp,
        fp=fp,
        fn=fn,
        ious=ious,
    )
    return pr_metrics


def batch_evaluate_segmentation(
    references: list[tuple[str, TimestampedOutputs]],
    predictions: list[tuple[str, TimestampedOutputs]],
    eval_config: EvaluationConfig,
) -> tuple[PrecisionRecallMetrics, list[PrecisionRecallMetrics]]:
    text_normalizer = eval_config.text_normalizer
    transcripts_must_match = eval_config.transcripts_must_match
    max_number_of_evals = eval_config.max_number_of_evals
    if len(references) != len(predictions):
        logger.error(
            f"There need to be as many reference entities (currently {len(references)}) "
            f"as prediction entities (currently {len(predictions)})."
        )
        return (
            PrecisionRecallMetrics(number_of_instances=0, tp=0, fp=0, fn=0, ious=[]),
            [],
        )
    seg_metrics_list = []
    eval_count = 0
    for ref, pred in zip(references, predictions):
        if not transcripts_must_match or text_normalizer(ref[0]) == text_normalizer(
            pred[0]
        ):
            eval_count += 1
            evaluation = evaluate_segmentation(
                references=ref[1],
                predictions=pred[1],
                eval_config=eval_config,
            )
            if transcripts_must_match and evaluation.fp != evaluation.fn:
                logger.error(
                    f"If transcripts match, also the number of false positives and false negatives must be equal."
                    f" This not being the case indicates a problem with the timestamps, therefore skipped sample {ref[0]}."
                )
                continue
            seg_metrics_list.append(evaluation)
        else:
            logger.debug(
                f"Normalized transcripts don't match. \n Reference: {text_normalizer(ref[0])} "
                f"\n Predicted: {text_normalizer(pred[0])}"
            )
        if max_number_of_evals != -1 and eval_count >= max_number_of_evals:
            break
    if not seg_metrics_list:
        return (
            PrecisionRecallMetrics(number_of_instances=0, tp=0, fp=0, fn=0, ious=[]),
            [],
        )
    return sum(seg_metrics_list), seg_metrics_list  # type: ignore


def convert_labels_json_to_list_of_TimestampedOutputs(
    reference_labels_json_path: Path,
    predicted_labels_json_path: Path,
) -> tuple[list[Any], list[Any], list[TimestampedOutputs], list[TimestampedOutputs]]:
    with reference_labels_json_path.open("r", encoding="utf-8") as labels_json:
        reference_labels = json.load(labels_json)
    with predicted_labels_json_path.open("r", encoding="utf-8") as labels_json:
        predicted_labels = json.load(labels_json)

    reference_dict = {label["audio"]: label for label in reference_labels}
    predicted_dict = {label["audio"]: label for label in predicted_labels}

    common_audio_keys = reference_dict.keys() & predicted_dict.keys()
    reference_labels_sorted = [reference_dict[key] for key in sorted(common_audio_keys)]
    predicted_labels_sorted = [predicted_dict[key] for key in sorted(common_audio_keys)]

    # Check if the audio keys align
    for ref_label, pred_label in zip(reference_labels_sorted, predicted_labels_sorted):
        if ref_label["audio"] != pred_label["audio"]:
            raise ValueError(
                f"Audio keys do not align: {ref_label['audio']} vs {pred_label['audio']}"
            )

    reference_transcripts, reference_timestamped_outputs = zip(
        *[
            (
                label["transcript"],
                convert_timestamps_from_labels_json_to_TimestampedOutput(
                    label["timings"]
                ),
            )
            for label in reference_labels_sorted
        ]
    )
    predicted_transcripts, predicted_timestamped_outputs = zip(
        *[
            (
                label["transcript"],
                convert_timestamps_from_labels_json_to_TimestampedOutput(
                    label["timings"]
                ),
            )
            for label in predicted_labels_sorted
        ]
    )
    reference_timestamped_outputs_with_transcripts = list(
        zip(reference_transcripts, reference_timestamped_outputs)
    )
    predicted_timestamped_outputs_with_transcripts = list(
        zip(predicted_transcripts, predicted_timestamped_outputs)
    )
    return (
        reference_timestamped_outputs_with_transcripts,
        predicted_timestamped_outputs_with_transcripts,
        list(reference_timestamped_outputs),
        list(predicted_timestamped_outputs),
    )


def evaluate_segmentation_from_label_json(
    reference_labels_json_path: Path,
    predicted_labels_json_path: Path,
    eval_config: EvaluationConfig,
) -> PrecisionRecallMetrics:
    (
        reference_timestamped_outputs,
        predicted_timestamped_outputs,
        _,
        _,
    ) = convert_labels_json_to_list_of_TimestampedOutputs(
        reference_labels_json_path=reference_labels_json_path,
        predicted_labels_json_path=predicted_labels_json_path,
    )
    # Evaluate segmentation
    evaluations, single_evaluations_list = batch_evaluate_segmentation(
        references=reference_timestamped_outputs,
        predictions=predicted_timestamped_outputs,
        eval_config=eval_config,
    )
    logger.info(evaluations)
    return evaluations


def get_normalized_timestamped_output_from_labels_json(
    eval_config: EvaluationConfig, prediction_path: Path, reference_path: Path
) -> tuple[list[TimestampedOutputs], list[TimestampedOutputs]]:
    (
        _,
        _,
        reference_timestamped_outputs,
        predicted_timestamped_outputs,
    ) = convert_labels_json_to_list_of_TimestampedOutputs(
        reference_labels_json_path=reference_path,
        predicted_labels_json_path=prediction_path,
    )
    references_list = []
    predictions_list = []
    if eval_config.pause_threshold > 0:
        for prediction in predicted_timestamped_outputs:
            prediction.adjust_pauses(
                split_threshold=eval_config.pause_threshold,
                pause_type=eval_config.pause_type,
            )
    if eval_config.do_normalize:
        for reference, prediction in zip(
            reference_timestamped_outputs, predicted_timestamped_outputs
        ):
            reference = reference.clean_timestamped_outputs(eval_config.text_normalizer)
            prediction = prediction.clean_timestamped_outputs(
                eval_config.text_normalizer
            )
            references_list.append(reference)
            predictions_list.append(prediction)
    return predictions_list, references_list


def analyze_time_shifts_and_durations(
    reference_path: Path, prediction_path: Path, eval_config: EvaluationConfig
) -> dict[str, Any]:
    (
        predictions_list,
        references_list,
    ) = get_normalized_timestamped_output_from_labels_json(
        eval_config, prediction_path, reference_path
    )
    collar = eval_config.collar

    start_shifts = []
    end_shifts = []
    duration_ratios = []
    for references, predictions in zip(references_list, predictions_list):
        for ref in references.entries:
            for pred in predictions.entries:
                if ref.word == pred.word and do_overlap(
                    ref.start, ref.end, pred.start, pred.end, collar
                ):
                    start_shifts.append(pred.start - ref.start)
                    end_shifts.append(pred.end - ref.end)
                    ref_duration = ref.end - ref.start
                    pred_duration = pred.end - pred.start
                    if ref_duration > 0:
                        duration_ratios.append(pred_duration / ref_duration)
                    break

    results = {
        "start_shift_mean": np.mean(start_shifts),
        "start_shift_std": np.std(start_shifts),
        "end_shift_mean": np.mean(end_shifts),
        "end_shift_std": np.std(end_shifts),
        "duration_ratio_mean": np.mean(duration_ratios),
        "duration_ratio_std": np.std(duration_ratios),
        "start_shifts": start_shifts,
        "end_shifts": end_shifts,
        "duration_ratios": duration_ratios,
    }

    # Plot histograms and boxplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    pause_str = (
        ""
        if eval_config.pause_threshold < 0
        else f" with pause_threshold of {eval_config.pause_threshold} seconds."
    )
    fig.suptitle(
        f"Comparison of {reference_path.name} (Ground Truth) vs {prediction_path.name} (Prediction)"
        + pause_str
    )

    for i, (title, data) in enumerate(
        [
            ("Start Time Shifts", start_shifts),
            ("End Time Shifts", end_shifts),
            ("Duration Ratios", duration_ratios),
        ]
    ):
        axs[i, 0].hist(data, bins=30)
        axs[i, 0].set_title(f"{title} - Histogram")
        axs[i, 0].set_xlabel("Seconds" if i < 2 else "Ratio")
        axs[i, 0].set_ylabel("Frequency")

        axs[i, 1].boxplot(data)
        axs[i, 1].set_title(f"{title} - Boxplot")
        axs[i, 1].set_ylabel("Seconds" if i < 2 else "Ratio")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pause_str = (
        ""
        if eval_config.pause_threshold < 0
        else f"_pause_threshold_{eval_config.pause_threshold}"
    )
    plot_path = (
        Path(__file__).parent
        / "data_preprocessing/plots/analysis/"
        / f"{reference_path.stem}_vs_{prediction_path.stem}{pause_str}.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    logger.info(f"Saved plot to {plot_path}.")
    return results


def advanced_evaluation(
    reference_path: Path, prediction_path: Path, eval_config: EvaluationConfig
) -> None:
    time_shift_duration_results = analyze_time_shifts_and_durations(
        reference_path, prediction_path, eval_config
    )
    logger.info("Time Shift and Duration Analysis:")
    logger.info(
        f"  Start Time Shift: Mean = {time_shift_duration_results['start_shift_mean']:.3f}s,"
        f" Std = {time_shift_duration_results['start_shift_std']:.3f}s"
    )
    logger.info(
        f"  End Time Shift: Mean = {time_shift_duration_results['end_shift_mean']:.3f}s,"
        f" Std = {time_shift_duration_results['end_shift_std']:.3f}s"
    )
    logger.info(
        f"  Duration Ratio: Mean = {time_shift_duration_results['duration_ratio_mean']:.3f},"
        f" Std = {time_shift_duration_results['duration_ratio_std']:.3f}"
    )
