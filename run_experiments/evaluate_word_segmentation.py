import copy
import json
from pathlib import Path
from pprint import pprint
from enum import Enum
import time
import re

from typing import Any, Callable, Optional, List, Dict
import sys

# Path to the 'src' folder inside the 'transformers' submodule
transformers_path = Path(__file__).parent.parent / "transformers/src"

# Add this path to the front of sys.path
sys.path.insert(0, str(transformers_path))

class AllowedCharacters(Enum):
    GERMAN_CASED: str = 'abcdefghijklmnopqrstuvwxyzäöüABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ'
    GERMAN_UNCASED: str = 'abcdefghijklmnopqrstuvwxyzäöü'
    ENGLISH_CASED: str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ENGLISH_UNCASED: str = 'abcdefghijklmnopqrstuvwxyz'
    
    
class SpecialCharacters(Enum):
    PUNCTUATION: str = '[,?.!;:\*()~"\-\',]'
    GERMAN_CASED: str = '[^a-zA-ZäöüÄÖÜß\\s]'  # all characters except a,...,z,A,...,Z,ä,ö,ü,ß,Ä,Ö,Ü
    GERMAN_UNCASED: str = '[^a-zäöüß\\s]'  # all characters except a,...,z,ä,ö,ü,ß
    ENGLISH_CASED: str = '[^a-zA-Z\\s]'  # all characters except a,...,z,A,...,Z
    ENGLISH_UNCASED: str = '[^a-z\\s]'  # all characters except a,...,z


class TextPreprocessor:
    """Class for preprocessing text files in a repeatable manner.

    Attributes:
        preprocessing_methods (Optional[List[str]]): A list of strings, specifying pre-processing methods. The
            order of these methods is important as they are executed sequentially.
    """
    def __init__(self, preprocessing_methods: Optional[List[str]] = None, tokens_not_to_be_preprocessed = [], tokens_mapping = {}, language: str="en"):
        self.preprocessing_methods = preprocessing_methods
        self.tokens_not_to_be_preprocessed = tokens_not_to_be_preprocessed
        self.tokens_mapping = tokens_mapping
        self.language = language

    def __call__(self, text: str) -> str:
        return self.preprocess(text)

    def replace_nums_in_text(self, text):
        numbers = re.findall(r'[0-9]+', text)
        for number in numbers:
            line = re.sub(number, num2words(number, self.language), text)
            text = line
        return text

    @staticmethod
    def token_to_lowercase_a_to_z_placeholder(token: str) -> str:
        timestamp = int(time.time())
        combined = f"{token}_{timestamp}"

        combined_sum = sum(ord(char) for char in combined)
        placeholder = ''
        while combined_sum > 0:
            combined_sum, remainder = divmod(combined_sum, 26)
            placeholder += chr(97 + remainder)
        return placeholder * 5

    @staticmethod
    def to_lower(text: str):
        return text.lower()

    @staticmethod
    def replace_special_characters(text, chars_to_replace: Optional[Dict[str, str]] = None):
        """Maps characters from keys to values."""
        if chars_to_replace is None:
            chars_to_replace = {"ß": "ss"}
        for to_replace, replace_with in chars_to_replace.items():
            text = re.sub(to_replace, replace_with, text)
        return text
    
    @staticmethod
    def phonemization_fixes(text: str, fixes: List[Callable]):
        for fix in fixes:
            text = fix(text)
        return text
    
    @staticmethod
    def replace_c_with_ze(sent: str = ''):
        patterns = [re.compile(r" c "), re.compile(r" c$"), re.compile(r"^c ")]
        output = re.sub(patterns[0],' ze ', sent)
        output = re.sub(patterns[1],' ze', output)
        output = re.sub(patterns[2],'ze ', output)
        return output

    @staticmethod
    def replace_pc_with_peze(sent: str = ''):
        patterns = [re.compile(r" pc "), re.compile(r"( pc)$"), re.compile(r"^(pc )")]
        output = re.sub(patterns[0],' peze ',sent)
        output = re.sub(patterns[1],' peze', output)
        output = re.sub(patterns[2],'peze ', output)
        return output

    @staticmethod
    def remove_unnecessary_spaces(text):
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def remove_all_spaces(text):
        return text.replace(" ", "")

    @staticmethod
    def remove_special_characters(text: str, characters_to_ignore: SpecialCharacters = SpecialCharacters.PUNCTUATION):
        """Removes all characters `characters_to_ignore`."""
        return re.sub(characters_to_ignore.value, '', text)

    def apply_text_processing_method(self, preprocessing_method: str, text: str):
        if preprocessing_method == "lower":
            return self.to_lower(text)
        elif preprocessing_method == "remove_punctuation":
            return self.remove_special_characters(text=text, characters_to_ignore=SpecialCharacters.PUNCTUATION)
        elif preprocessing_method == "replace_special_chars":
            return self.replace_special_characters(text=text)
        elif preprocessing_method == "remove_non_german_chars":
            return self.remove_special_characters(text=text, characters_to_ignore=SpecialCharacters.GERMAN_UNCASED)
        elif preprocessing_method == "remove_non_english_chars":
            return self.remove_special_characters(text=text, characters_to_ignore=SpecialCharacters.ENGLISH_UNCASED) 
        elif preprocessing_method == "replace_digits_with_words":
            return self.replace_nums_in_text(text=text)
        elif preprocessing_method == "phonemization_fixes":
            return self.phonemization_fixes(text=text,
                                            fixes=[self.replace_c_with_ze,
                                            self.replace_pc_with_peze])
        elif preprocessing_method == "remove_unnecessary_spaces":
            return self.remove_unnecessary_spaces(text)
        elif preprocessing_method == "remove_all_spaces":
            return self.remove_all_spaces(text)
        else:
            raise NotImplementedError(f"{preprocessing_method} is not a valid pre-processing method.")

    def preprocess(self, text: str) -> str:
        """Applies the `preprocessing_methods` sequentially to `text`."""
        token_to_placeholder = {token: self.token_to_lowercase_a_to_z_placeholder(token) for token
                                in self.tokens_not_to_be_preprocessed}
        text = str(text)
        for token, placeholder in token_to_placeholder.items():
            text = text.replace(token, placeholder)
        for preprocessing_method in self.preprocessing_methods:
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
        "remove_all_spaces",
    ],
    tokens_not_to_be_preprocessed=["[UH]", "[UM]"],
    language="en",
)


class TimestampedOutput:
    """Represents a word with timestamps and an optional type."""

    def __init__(self, word: str, start: float, end: float, word_type: str | None = ""):
        self.word = word
        self.start = start
        self.end = end
        self.word_type = word_type

    def to_dict(self) -> dict[str, Any]:
        """Converts the timestamped output to a dictionary."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "type": self.word_type,
        }

    @classmethod
    def from_dict(cls, result: dict[str, Any]):
        return cls(
            word=result["word"],
            start=result["start"],
            end=result["end"],
            word_type=result.get("type", ""),
        )

    def __str__(self):
        """String representation of the TimestampedOutput."""
        return f'Word: "{self.word}", Start: {self.start}, End: {self.end}, Type: {self.word_type or "N/A"}'


class TimestampedOutputs:
    """Collection of timestamped word outputs."""

    def __init__(self, entries: list[TimestampedOutput]):
        self.entries = entries

    def to_json(self) -> list[dict[str, Any]]:
        """Converts timestamped outputs to a JSON-serializable list."""
        return [entry.to_dict() for entry in self.entries]

    @classmethod
    def from_dict(cls, result: list[dict[str, Any]]):
        entries = []
        for entry in result:
            tso = TimestampedOutput.from_dict(entry)
            entries.append(tso)
        return cls(entries=entries)

    def __str__(self):
        """String representation of the TimestampedOutputs."""
        return "\n".join(str(entry) for entry in self.entries)

    def clean_timestamped_outputs(
        self,
        text_normalizer: Callable = DEFAULT_NORMALIZER,
    ):
        result = []
        entries = copy.deepcopy(self.entries)
        for entry in entries:
            normalized_word = text_normalizer(entry.word)
            entry.word = normalized_word
            if normalized_word and len(normalized_word):
                result.append(entry)
        result = TimestampedOutputs(result)
        return result
    
    def adjust_pauses(self, split_threshold: float = 0.06):
        """Adjust the timing of pauses that are spaces (' ') and exceed the threshold."""
        for i in range(1, len(self.entries) - 1):
            entry = self.entries[i]
            if entry.word == ' ':
                duration = entry.end - entry.start
                if duration > split_threshold:
                    distribute = split_threshold / 2
                else:
                    distribute = duration/2
                # Check the previous and next words
                if self.entries[i - 1].word != ' ':
                    self.entries[i - 1].end += distribute
                if self.entries[i + 1].word != ' ':
                    self.entries[i + 1].start -= distribute

                # Update the current pause's timing
                entry.start += distribute
                entry.end -= distribute

                        
        


def convert_timestamps_from_transformers_pipe_to_TimestampedOutput(
    timestamps: list[dict[str, Any]]
):
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
        updated_list.append({'word': ' ', 'start': word1['end'], 'end': word2['start']})
    if len(converted_list) >=1:
        updated_list.append(converted_list[-1])
        
    return TimestampedOutputs.from_dict(updated_list)


def convert_timestamps_from_labels_json_to_TimestampedOutput(
    timestamps: list[dict[str, Any]]
):
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


def do_overlap(start1, end1, start2, end2, collar=0.2):
    overlap = abs(start1 - start2) <= collar and abs(end1 - end2) <= collar
    return overlap


def calculate_iou(start1, end1, start2, end2):
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


def get_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall


class PrecisionRecallMetrics:
    def __init__(
        self,
        tp: int,
        fp: int,
        fn: int,
        ious: list | None = None,
    ):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.ious = ious

    @property
    def precision(self):
        return self.get_precision_recall()[0]

    @property
    def recall(self):
        return self.get_precision_recall()[1]

    @property
    def f1_score(self):
        return self.calculate_f1_score()

    @property
    def avg_iou(self):
        return sum(self.ious) / len(self.ious) if len(self.ious) else 0.0

    def __add__(self, other):
        new_tp = self.tp + other.tp
        new_fp = self.fp + other.fp
        new_fn = self.fn + other.fn

        new_ious = self.ious + other.ious
        return PrecisionRecallMetrics(new_tp, new_fp, new_fn, new_ious)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_precision_recall(self):
        tp = self.tp
        fp = self.fp
        fn = self.fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        return precision, recall

    def calculate_f1_score(self):
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
    collar: float,
    text_normalizer: Callable = DEFAULT_NORMALIZER,
) -> PrecisionRecallMetrics:
    references = references.clean_timestamped_outputs(text_normalizer)
    predictions = predictions.clean_timestamped_outputs(text_normalizer)
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

            if gt_word == pred_word and do_overlap(
                gt_start, gt_end, pred_start, pred_end, collar=collar
            ):
                if not prediction_matched[
                    i
                ]:  # Ensure each prediction is only counted once
                    tp += 1
                    prediction_matched[i] = True
                    found_match = True
                    ious.append(calculate_iou(gt_start, gt_end, pred_start, pred_end))

                    break  # Move to the next ground truth word after finding a match

        if not found_match:
            fn += 1

    # Count False Positives
    fp = prediction_matched.count(False)

    pr_metrics = PrecisionRecallMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        ious=ious,
    )
    return pr_metrics


def batch_evaluate_segmentation(
    references: list[tuple[str, TimestampedOutputs]],
    predictions: list[tuple[str, TimestampedOutputs]],
    collar: float,
    transcripts_must_match: bool = True,
    text_normalizer: Callable = DEFAULT_NORMALIZER,
    max_number_of_evals: int = -1,
) -> PrecisionRecallMetrics:
    if len(references) != len(predictions):
        print(
            f"There need to be as many reference entities (currently {len(references)}) "
            f"as prediction entities (currently {len(predictions)})."
        )
        return PrecisionRecallMetrics(tp=0, fp=0, fn=0, ious=[]), []
    seg_metrics_list = []
    eval_count = 0
    for ref, pred in zip(references, predictions):
        # print(f"Normalized gt transcript  : {text_normalizer(ref[0])}")
        # print(f"Normalized pred transcript: {text_normalizer(pred[0])}")
        if not transcripts_must_match or text_normalizer(ref[0]) == text_normalizer(
            pred[0]
        ):
            eval_count += 1
            seg_metrics_list.append(
                evaluate_segmentation(
                    references=ref[1],
                    predictions=pred[1],
                    collar=collar,
                    text_normalizer=text_normalizer,
                )
            )
        if max_number_of_evals != -1 and eval_count >= max_number_of_evals:
            break
    if not seg_metrics_list:
        return PrecisionRecallMetrics(tp=0, fp=0, fn=0, ious=[]), []
    return sum(seg_metrics_list), seg_metrics_list

