import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torchaudio
from loguru import logger


def plot_transcript_on_spectrogram(
    audio_path: Path,
    transcripts_with_timestamps: list[dict[str, Any]],
    output_dir: Path = Path(__file__).parent,
) -> None:
    sample, sample_rate = torchaudio.load(audio_path)
    title = f"Audio: {audio_path.name}"

    fig, axes = plt.subplots(
        len(transcripts_with_timestamps),
        1,
        figsize=(10, len(transcripts_with_timestamps) * 4),
    )
    fig.suptitle(title, fontsize=14)

    if len(transcripts_with_timestamps) == 1:
        axes = [axes]

    max_end_time = max(
        max(timestamp["endtime"] for timestamp in transcript_with_timestamps["timings"])
        for transcript_with_timestamps in transcripts_with_timestamps
    )

    for idx, timestamped_transcript in enumerate(transcripts_with_timestamps):
        ax = axes[idx]
        Pxx, freqs, bins, im = ax.specgram(
            sample.squeeze(), NFFT=1024, Fs=sample_rate, noverlap=512, cmap="viridis"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim(0, max_end_time)
        ax.set_title(f"{timestamped_transcript['model']}")
        previous_end_time = 0
        for timestamp in timestamped_transcript["timings"]:
            start = timestamp["starttime"]
            end = timestamp["endtime"]
            word = timestamp["word"]

            # Highlight pauses
            if start > previous_end_time:
                ax.axvspan(
                    previous_end_time, start, color="white", hatch="/", alpha=0.5
                )

            ax.text(
                (start + end) / 2,
                sample_rate / 4,
                word,
                horizontalalignment="center",
                color="white",
                fontsize=10,
                weight="bold",
            )
            ax.axvline(x=start, color="r", linestyle="--")
            ax.axvline(x=end, color="r", linestyle="--")

            previous_end_time = end

        # Highlight the final pause if any
        if previous_end_time < max_end_time:
            ax.axvspan(
                previous_end_time, max_end_time, color="gray", alpha=0.3, hatch="/"
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "plots" / f"{audio_path.stem}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    logger.info(f"Saved plot to {plot_path}.")


def filter_common_audio_labels(
    json_paths: list[Path],
) -> dict[str, Any]:
    # Load JSON data from each path
    all_labels = []
    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            all_labels.append(json.load(f))

    # Find common audio keys
    common_audio_keys = set(entry["audio"] for entry in all_labels[0])
    for labels in all_labels[1:]:
        audio_keys = set(entry["audio"] for entry in labels)
        common_audio_keys &= audio_keys
    base_data_dir = json_paths[0].parent
    # Filter entries by common audio keys
    filtered_labels: dict[str, Any] = {
        str(base_data_dir / audio_key): [] for audio_key in common_audio_keys
    }
    for labels in all_labels:
        for entry in labels:
            audio_key = entry["audio"]
            if audio_key in common_audio_keys:
                if not entry["timings"]:
                    logger.debug(
                        f"No timestamps for audio file {audio_key} for model {entry['dataset_name']}"
                    )
                else:
                    filtered_labels[str(base_data_dir / entry["audio"])].append(
                        {"timings": entry["timings"], "model": entry["dataset_name"]}
                    )

    return filtered_labels


def main() -> None:
    base_dir = Path(__file__).parent.parent / "data/cv_17_manually_labelled"

    labels_files = [
        "labels.json",
    ]
    labels_paths = [base_dir / label_file for label_file in labels_files]
    common_labels = filter_common_audio_labels(labels_paths)
    for index, (audio_path, word_with_timestamps) in enumerate(common_labels.items()):
        plot_transcript_on_spectrogram(
            audio_path=Path(audio_path),
            transcripts_with_timestamps=word_with_timestamps,
        )


if __name__ == "__main__":
    main()
