import copy
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import string
import sys
import torch
import torchaudio
import tqdm
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    add_gaussian_noise_to_audio,
    add_score_metrics,
    adjust_pauses,
    common_indexes,
    configure_model,
    configure_model_generation,
    find_matching_indexes,
    get_alignment_head_generator,
    get_processor_config_genConfig,
    get_mixing_scale,
    load_labels,
    prepare_ground_truth,
    process_audio_files,
    remove_punctuation,
    setup_pipeline,
    top_heads_by_f1,
)
from pathlib import Path
from evaluate_word_segmentation import (
    convert_timestamps_from_labels_json_to_TimestampedOutput,
    convert_timestamps_from_transformers_pipe_to_TimestampedOutput,
    batch_evaluate_segmentation,
)
from crisper_whisper import WhisperForConditionalGenerationWithAttentionLoss

from speech_recognition import (
    WhisperXModel,
    WhisperTimestamped,
    ModelConfig,
    transcribe_speech_files,
)
from transformers import (
    AutoConfig,
    AutoProcessor,
    WhisperProcessor,
    AutoModelForSpeechSeq2Seq,
    GenerationConfig,
    pipeline,
)

from typing import Dict, Generator, List
from loguru import logger


def main():
    experiment_config = {
        "models": [
            {"name": "large-v3", "path": "openai/whisper-large-v3"},
            {"name": "large-v2", "path": "openai/whisper-large-v2"},
            # {"name": "crisperWhisper++", "path": "/home/azureuser/laurin/code/research/output/crisper_whisper_timestamp_finetuned"}
        ],
        "output_path": "experiments",
        "dataset_base_path": "/home/azureuser/data/english_speech",
        "head_selection_dataset": "timit",
        "head_selection_collar": 0.1,
        "head_selection_num_samples": 100,
        "experiment_num_samples": 500,
        "test_datasets": [
            "synthetic_no_fillers_long_pauses",
            "timit",
            "ami_hf",
            "cv_14",
        ],
        "device": "cuda:0",
        "median_filter_widths": [1, 3, 5, 7, 9],
        "num_heads": list(range(1, 15)),
        "add_noise": [False],
        "transcripts_must_match": True,
        "pause_thresholds": np.linspace(0, 0.2, 5),
        "collars": [float(x) / 20 for x in range(1, 21)],
    }

    # Create output directory if it doesn't exist
    os.makedirs(experiment_config["output_path"], exist_ok=True)

    # In this section, we extract the F1 segmentation Scores on a subset of a dataset for each individiual head of the decoder.
    results = []
    for model_config in experiment_config["models"]:
        model_name = model_config["name"]
        model_path = model_config["path"]
        model = configure_model(model_path, experiment_config["device"])
        processor, config, generation_config = get_processor_config_genConfig(
            model_path
        )
        alignment_head_generator = get_alignment_head_generator(32, 20)

        for alignment_heads in tqdm.tqdm(
            alignment_head_generator, desc=f"Processing {model_name}"
        ):
            configure_model_generation(model, model_name, processor, alignment_heads, 7)
            labels = load_labels(
                experiment_config["dataset_base_path"],
                experiment_config["head_selection_dataset"],
                "test",
                experiment_config["head_selection_num_samples"],
            )
            ground_truth_transcripts_and_timestamps = prepare_ground_truth(labels)
            asr_pipeline = setup_pipeline(model, processor, experiment_config["device"])

            audio_paths = [
                os.path.join(
                    experiment_config["dataset_base_path"],
                    experiment_config["head_selection_dataset"],
                    label["audio"],
                )
                for label in labels
            ]
            try:
                predictions, success_ids = process_audio_files(
                    audio_paths, asr_pipeline
                )
                ground_truth_transcripts_and_timestamps = [
                    ground_truth_transcripts_and_timestamps[i] for i in success_ids
                ]

                predicted_transcripts_and_timestamps = [
                    (
                        prediction["text"],
                        convert_timestamps_from_transformers_pipe_to_TimestampedOutput(
                            prediction["chunks"]
                        ),
                    )
                    for prediction in predictions
                ]

                new_predictions = adjust_pauses(predicted_transcripts_and_timestamps)
                seg_metrics, _ = batch_evaluate_segmentation(
                    ground_truth_transcripts_and_timestamps,
                    new_predictions,
                    collar=experiment_config["head_selection_collar"],
                    transcripts_must_match=experiment_config["transcripts_must_match"],
                )

                results.append(
                    {
                        "model": model_name,
                        "heads": alignment_heads,
                        f'F1_collar.{experiment_config["head_selection_collar"]}': seg_metrics.f1_score,
                        f'avg_iou_collar.{experiment_config["head_selection_collar"]}': seg_metrics.avg_iou,
                    }
                )
            except Exception as e:
                logger.info(e)

    with open(
        os.path.join(experiment_config["output_path"], "head_results.json"), "w"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # In this section, we will calculate the results when greedily adding "the best" heads from the previous run.
    # Further we will vary various hyperparameters that might have a influence on the segmentation so we are able to perform ablation studies.
    with open(
        os.path.join(experiment_config["output_path"], "head_results.json"), "r"
    ) as f:
        result = json.load(f)
    head_df = pd.DataFrame(result)
    # Set logging level to CRITICAL to suppress all other log messages
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    results = []
    output = []
    for model_config in tqdm.tqdm(experiment_config["models"]):
        logger.info(f"working on {model_config}")
        model_name = model_config["name"]
        model_path = model_config["path"]
        model = configure_model(model_path, experiment_config["device"])
        processor, config, generation_config = get_processor_config_genConfig(
            model_path
        )

        for dataset in experiment_config["test_datasets"]:
            logger.info(f"working on {dataset}")
            for add_noise in experiment_config["add_noise"]:
                for median_filter_width in experiment_config["median_filter_widths"]:
                    for num_heads in experiment_config["num_heads"]:
                        for legacy in [True, False]:
                            top_heads_dict = top_heads_by_f1(
                                head_df,
                                num_heads,
                                experiment_config["head_selection_collar"],
                            )
                            alignment_heads = top_heads_dict[model_name]
                            configure_model_generation(
                                model,
                                model_name,
                                processor,
                                alignment_heads,
                                median_filter_width,
                            )
                            labels = load_labels(
                                experiment_config["dataset_base_path"],
                                dataset,
                                split="test",
                                limit=experiment_config["experiment_num_samples"],
                            )
                            model.generation_config.legacy = legacy
                            ground_truth_transcripts_and_timestamps = (
                                prepare_ground_truth(labels)
                            )
                            asr_pipeline = setup_pipeline(
                                model, processor, experiment_config["device"]
                            )
                            audio_paths = [
                                os.path.join(
                                    experiment_config["dataset_base_path"],
                                    dataset,
                                    label["audio"],
                                )
                                for label in labels
                            ]
                            if add_noise:
                                audio_paths = [
                                    add_gaussian_noise_to_audio(
                                        path, random.randint(1, 8)
                                    )
                                    for path in audio_paths
                                ]

                            predictions, success_ids = process_audio_files(
                                audio_paths, asr_pipeline
                            )
                            ground_truth_transcripts_and_timestamps = [
                                ground_truth_transcripts_and_timestamps[i]
                                for i in success_ids
                            ]

                            predicted_transcripts_and_timestamps = [
                                (
                                    prediction["text"],
                                    convert_timestamps_from_transformers_pipe_to_TimestampedOutput(
                                        prediction["chunks"]
                                    ),
                                )
                                for prediction in predictions
                            ]
                            for threshold in experiment_config["pause_thresholds"]:
                                new_predictions = adjust_pauses(
                                    predicted_transcripts_and_timestamps,
                                    threshold=threshold,
                                )
                                for collar in experiment_config["collars"]:
                                    (
                                        seg_metrics,
                                        seg_metrics_list,
                                    ) = batch_evaluate_segmentation(
                                        ground_truth_transcripts_and_timestamps,
                                        new_predictions,
                                        collar=collar,
                                        transcripts_must_match=experiment_config[
                                            "transcripts_must_match"
                                        ],
                                    )
                                    addendum = "_noise" if add_noise else ""
                                    model_name_addendum = (
                                        "legacy" if legacy else "regular"
                                    )
                                    output.append(
                                        {
                                            "Threshold": threshold,
                                            "Collar": collar,
                                            "Model": f"{model_name}_{model_name_addendum}",
                                            "MedianFilterWidth": median_filter_width,
                                            "Dataset": dataset + addendum,
                                            "Recall": seg_metrics.recall,
                                            "Precision": seg_metrics.precision,
                                            "F1 Score": seg_metrics.f1_score,
                                            "Avg IOU": seg_metrics.avg_iou,
                                            "num_heads": num_heads,
                                            "metrics": seg_metrics_list,
                                        }
                                    )

    with open(
        os.path.join(experiment_config["output_path"], "legacy_vs_hf_ablations.pickle"),
        "wb",
    ) as f:
        pickle.dump(output, f)

    baseline_output = []
    for i, model_config in enumerate(
        [ModelConfig.TIMESTAMPED_WHISPER, ModelConfig.WHISPER_X]
    ):
        if i == 1:
            model = WhisperXModel(**model_config)
        elif i == 0:
            model = WhisperTimestamped(**model_config)
        for dataset in experiment_config["test_datasets"]:
            for add_noise in experiment_config["add_noise"]:
                labels = load_labels(
                    experiment_config["dataset_base_path"],
                    dataset,
                    split="test",
                    limit=experiment_config["experiment_num_samples"],
                )
                audio_paths = [
                    os.path.join(
                        experiment_config["dataset_base_path"], dataset, label["audio"]
                    )
                    for label in labels
                ]
                if add_noise:
                    audio_paths = [
                        add_gaussian_noise_to_audio(path, random.randint(1, 8))
                        for path in audio_paths
                    ]

                outputs, error_ids = transcribe_speech_files(
                    model=model, dataset_name=dataset, audio_paths=audio_paths
                )

                timestamped_outputs = [
                    (output.prediction_str, output.timestamped_outputs)
                    for output in outputs
                ]
                labels = [
                    label
                    for label in labels
                    if os.path.join(
                        experiment_config["dataset_base_path"], dataset, label["audio"]
                    )
                    not in error_ids
                ]
                ground_truth_transcripts_and_timestamps = prepare_ground_truth(labels)
                for collar in experiment_config["collars"]:
                    seg_metrics, seg_metrics_list = batch_evaluate_segmentation(
                        ground_truth_transcripts_and_timestamps,
                        timestamped_outputs,
                        collar=collar,
                        transcripts_must_match=experiment_config[
                            "transcripts_must_match"
                        ],
                    )
                    addendum = "_noise" if add_noise else ""
                    baseline_output.append(
                        {
                            "Threshold": None,
                            "Collar": collar,
                            "Model": "WhisperX" if i == 1 else "WhisperTimestamped",
                            "MedianFilterWidth": None,
                            "Dataset": dataset + addendum,
                            "Recall": seg_metrics.recall,
                            "Precision": seg_metrics.precision,
                            "F1 Score": seg_metrics.f1_score,
                            "Avg IOU": seg_metrics.avg_iou,
                            "num_heads": None,
                            "predictions": outputs,
                            "gts": ground_truth_transcripts_and_timestamps,
                            "metrics": seg_metrics_list,
                        }
                    )

    with open(
        os.path.join(experiment_config["output_path"], "baseline_ablations.pickle"),
        "wb",
    ) as f:
        pickle.dump(baseline_output, f)

    baseline_df = pd.DataFrame(baseline_output)
    df = baseline_df
    # Filter dataframe for Collar == 0.20
    filtered_df = df[df["Collar"] == 0.20]

    # Group by Model and Dataset, then find the maximum F1 Score and corresponding Avg IOU
    result_df = (
        filtered_df.groupby(["Model", "Dataset"])
        .apply(lambda x: x.loc[x["F1 Score"].idxmax(), ["F1 Score", "Avg IOU"]])
        .reset_index()
    )

    pivoted_df = result_df.pivot(
        index="Dataset", columns="Model", values=["F1 Score", "Avg IOU"]
    )

    # Flatten the MultiIndex columns
    pivoted_df.columns = ["_".join(col).strip() for col in pivoted_df.columns.values]
    pivoted_df.reset_index(inplace=True)

    logger.info(pivoted_df)

    with open(
        os.path.join(experiment_config["output_path"], "full_ablations.pickle"), "rb"
    ) as f:
        output = pickle.load(f)

    visualization_config = {
        "models_to_compare": ["large-v3", "crisperWhisper++", "large-v2"],
        "output_path": "plots",
        "visualization_datasets": [
            "timit",
            "ami_hf",
            "cv_14",
            "synthetic_no_fillers_long_pauses",
        ],
    }

    # Create output directory if it doesn't exist
    Path(visualization_config["output_path"]).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(output)
    logger.info(df)
    # Apply the function to each row in the DataFrame for each dataset
    df["matching_indexes"] = df.apply(find_matching_indexes, axis=1)
    # Group by 'Model' and 'Dataset' and apply the common_indexes function
    common_idx_df = (
        df.groupby(["Model", "Dataset"])["matching_indexes"]
        .agg(common_indexes)
        .reset_index()
    )

    # Merge the common indexes back into the original DataFrame
    df = pd.merge(df, common_idx_df, on=["Model", "Dataset"], suffixes=("", "_common"))
    df["F1 Score clean"] = df.apply(
        lambda row: add_score_metrics(row, "f1_score"), axis=1
    )
    df["Avg IOU clean"] = df.apply(
        lambda row: add_score_metrics(row, "avg_iou"), axis=1
    )
    for dataset in visualization_config["visualization_datasets"]:
        collar = 0.1
        threshold = 0.1  # threshold for splitting pauses
        filtered_df = df[df["Collar"] == collar]
        filtered_df = filtered_df[filtered_df["MedianFilterWidth"] == 3]
        filtered_df = filtered_df[filtered_df["Threshold"] == threshold]
        filtered_df = filtered_df[
            filtered_df["Model"].isin(visualization_config["models_to_compare"])
        ]
        filtered_df = filtered_df[filtered_df["Dataset"] == dataset]
        # Set the plotting style
        sns.set(style="whitegrid")

        # Initialize the matplotlib figure
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Define colors and markers for better distinction
        color_f1 = "tab:blue"
        color_iou = "tab:green"
        marker_f1 = "o"
        marker_iou = "X"

        # Plotting F1 Score on the first y-axis
        f1_lines = sns.lineplot(
            data=filtered_df,
            x="num_heads",
            y="F1 Score",
            hue="Model",
            marker=marker_f1,
            ax=ax1,
            palette="Blues",
        )
        ax1.set_xlabel("Number of Heads", fontsize=14)
        ax1.set_ylabel("F1 Score", fontsize=14, color=color_f1)
        ax1.tick_params(axis="y", labelcolor=color_f1)
        ax1.legend(title="Model (F1 Score)", bbox_to_anchor=(1.12, 1), loc="upper left")

        ax1.legend(title="F1 Score", bbox_to_anchor=(1.12, 1), loc="upper left")

        # Create a second y-axis for Avg IOU, sharing the same x-axis
        ax2 = ax1.twinx()
        iou_lines = sns.lineplot(
            data=filtered_df,
            x="num_heads",
            y="Avg IOU",
            hue="Model",
            marker=marker_iou,
            ax=ax2,
            palette="Greens",
        )
        ax2.set_ylabel("Avg IOU", fontsize=14, color=color_iou)
        ax2.tick_params(axis="y", labelcolor=color_iou)
        ax2.legend(
            title="Model (Avg IOU)", bbox_to_anchor=(1.12, 0.85), loc="upper left"
        )
        ax2.legend(title="Avg IOU", bbox_to_anchor=(1.12, 0.85), loc="upper left")

        # Title and layout adjustment
        fig.suptitle(
            f"Comparison of F1 Score and Avg IOU vs. Number of greedily selected Heads at Collar={collar} for Dataset: {dataset}",
            fontsize=16,
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                visualization_config["output_path"],
                f"Average_F1_IOU_vs_number_of_heads_collar_{collar}_dataset_{dataset}.png",
            )
        )
        # Show the plot
        plt.show()

    # Assuming your DataFrame is named df
    # Calculate the mean F1 Score and Avg IOU for each Model and num_heads
    collar = 0.1
    filtered_df = df[df["Collar"] == collar]
    filtered_df = filtered_df[
        filtered_df["Model"].isin(visualization_config["models_to_compare"])
    ]
    filtered_df = filtered_df[filtered_df["MedianFilterWidth"] == 3]
    filtered_df = filtered_df[
        filtered_df["Threshold"] == 0.1
    ]  # threshold for splitting pauses
    grouped_df = (
        filtered_df.groupby(["Model", "num_heads"])
        .agg(
            {
                "F1 Score": "mean",
                "Avg IOU": "mean",
                "F1 Score": "mean",
                "Avg IOU": "mean",
            }
        )
        .reset_index()
    )

    # Set the plotting style
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Colors for the two metrics
    colors = ["tab:blue", "tab:green"]

    # Plot F1 Score
    sns.lineplot(
        data=grouped_df,
        x="num_heads",
        y="F1 Score",
        hue="Model",
        ax=ax1,
        palette="Blues",
        marker="o",
    )
    ax1.set_xlabel("Number of Heads", fontsize=14)
    ax1.set_ylabel("Average F1 Score", fontsize=14, color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    # Create a second y-axis for Avg IOU
    ax2 = ax1.twinx()
    sns.lineplot(
        data=grouped_df,
        x="num_heads",
        y="Avg IOU",
        hue="Model",
        ax=ax2,
        palette="Greens",
        marker="X",
    )
    ax2.set_ylabel("Average Avg IOU", fontsize=14, color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    # Add legends
    ax1.legend(title="Model (F1 Score)", bbox_to_anchor=(1.12, 1), loc="upper left")
    ax2.legend(title="Model (Avg IOU)", bbox_to_anchor=(1.12, 0.85), loc="upper left")

    # Title and layout adjustment
    fig.suptitle(
        f"Average F1 Score and Avg IOU vs. Number of Heads Across Datasets for Collar: {collar}",
        fontsize=16,
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            visualization_config["output_path"],
            f"Average_F1_IOU_vs_number_of_heads_collar_{collar}.png",
        )
    )
    # Show the plot
    plt.show()

    # Assuming your DataFrame is named df
    # Calculate the mean F1 Score and Avg IOU for each Model and num_heads
    for dataset in visualization_config["visualization_datasets"]:
        # Feel free to play around with these values
        collar = 0.1
        filtered_df = df[df["Collar"] == collar]
        filtered_df = filtered_df[filtered_df["Dataset"] == dataset]
        filtered_df = filtered_df[filtered_df["MedianFilterWidth"] == 3]
        filtered_df = filtered_df[filtered_df["num_heads"] == 3]
        filtered_df = filtered_df[filtered_df["Model"].isin(["crisperWhisper++"])]
        grouped_df = (
            filtered_df.groupby(["Model", "Threshold"])
            .agg(
                {
                    "F1 Score": "mean",
                    "Avg IOU": "mean",
                }
            )
            .reset_index()
        )

        # Set the plotting style
        sns.set(style="whitegrid")

        # Initialize the matplotlib figure
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Colors for the two metrics
        colors = ["tab:blue", "tab:green"]

        # Plot F1 Score
        sns.lineplot(
            data=grouped_df,
            x="Threshold",
            y="F1 Score",
            hue="Model",
            ax=ax1,
            palette="Blues",
            marker="o",
        )
        ax1.set_xlabel("Pause Splitting Threshold", fontsize=14)
        ax1.set_ylabel("Average F1 Score", fontsize=14, color=colors[0])
        ax1.tick_params(axis="y", labelcolor=colors[0])

        # Create a second y-axis for Avg IOU
        ax2 = ax1.twinx()
        sns.lineplot(
            data=grouped_df,
            x="Threshold",
            y="Avg IOU",
            hue="Model",
            ax=ax2,
            palette="Greens",
            marker="X",
        )
        ax2.set_ylabel("Average Avg IOU", fontsize=14, color=colors[1])
        ax2.tick_params(axis="y", labelcolor=colors[1])

        # Add legends
        ax1.legend(title="Model (F1 Score)", bbox_to_anchor=(1.12, 1), loc="upper left")
        ax2.legend(
            title="Model (Avg IOU)", bbox_to_anchor=(1.12, 0.85), loc="upper left"
        )

        # Title and layout adjustment
        fig.suptitle(
            f"Average F1 Score and Avg IOU vs. Threshold for Dataset: {dataset} for Collar: {collar}",
            fontsize=16,
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                visualization_config["output_path"],
                f"Average_F1_IOU_vs_pause_splitting_threshold_dataset_{dataset}_collar_{collar}.png",
            )
        )

        # Show the plot
        plt.show()

    for dataset in visualization_config["visualization_datasets"]:
        # Define thresholds for each model (customize this as needed)
        filtered_df = df[df["Threshold"] == 0.1]
        filtered_df = filtered_df[filtered_df["num_heads"] == 3]
        filtered_df = filtered_df[filtered_df["Dataset"] == dataset]
        filtered_df = filtered_df[filtered_df["MedianFilterWidth"] == 3]
        fig, ax = plt.subplots()
        for model, group in filtered_df.groupby("Model"):
            ax.plot(group["Collar"], group["F1 Score"], label=model, marker="o")

        ax.set_xlabel("Collar")
        ax.set_ylabel("F1 Score Clean")
        ax.set_title(f"F1 Score by Collar for Each Model for Dataset: {dataset}")
        ax.legend(title="Model")

        plt.savefig(
            os.path.join(
                visualization_config["output_path"],
                f"Average_F1_vs_collar_dataset_{dataset}.png",
            )
        )

        plt.show()

    for dataset in visualization_config["visualization_datasets"]:
        collar = 0.1
        filtered_df = df[df["Threshold"] == 0.1]
        filtered_df = filtered_df[filtered_df["num_heads"] == 3]
        filtered_df = filtered_df[filtered_df["Dataset"] == dataset]
        filtered_df = filtered_df[filtered_df["Collar"] == collar]
        fig, ax = plt.subplots()
        for model, group in filtered_df.groupby("Model"):
            ax.plot(
                group["MedianFilterWidth"], group["F1 Score"], label=model, marker="o"
            )

        ax.set_xlabel("Median Filter Width")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"F1 Score by Collar for Each Model for Dataset: {dataset}")
        ax.legend(title="Model")
        plt.savefig(
            os.path.join(
                visualization_config["output_path"],
                f"Average_F1_vs_median_filter_width_dataset_{dataset}_collar_{collar}.png",
            )
        )

        plt.show()


if __name__ == "__main__":
    main()
