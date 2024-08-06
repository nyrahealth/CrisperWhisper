import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any

import cattrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from evaluate_word_segmentation import (
    EvaluationConfig,
    batch_evaluate_segmentation,
    convert_timestamps_from_transformers_pipe_to_TimestampedOutput,
)
from loguru import logger
from speech_recognition import (
    ModelConfig,
    WhisperTimestamped,
    WhisperXModel,
    transcribe_speech_files,
)
from tqdm import tqdm
from utils import (
    add_gaussian_noise_to_audio,
    add_score_metrics,
    adjust_pauses,
    common_indexes,
    configure_model,
    configure_model_generation,
    find_matching_indexes,
    get_alignment_head_generator,
    get_processor_config_and_genconfig,
    load_labels,
    prepare_ground_truth,
    process_audio_files,
    setup_pipeline,
    top_heads_by_f1,
)

converter = cattrs.Converter()


def load_experiment_config() -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    return {
        "models": [
            # {"name": "large-v2", "path": "openai/whisper-large-v2"},
            {"name": "large-v3", "path": "openai/whisper-large-v3"},
            # {
            #     "name": "crisperWhisper++",
            #     "path": str(
            #         Path(__file__).parent.parent
            #         / "model/crisper_whisper_timestamp_finetuned"
            #     ),
            # },
        ],
        "output_path": "experiments",
        "dataset_base_path": Path(__file__).parent.parent / "data",
        "head_selection_dataset": "timit",
        "head_selection_collar": 0.1,
        "head_selection_num_samples": 2,
        "head_selection_num_heads": 32,
        "head_selection_num_layers": 20,
        "experiment_num_samples": 2,
        "test_datasets": [
            "timit",
        ],
        "device": device,
        "median_filter_widths": [3],
        "num_heads": [1],
        "add_noise": [False],
        "transcripts_must_match": True,
        "pause_thresholds": np.linspace(0, 0.2, 2),
        "collars": [0.1],
    }


def process_alignment_heads(config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    eval_config = EvaluationConfig(  # type: ignore
        collar=config["head_selection_collar"],
        transcripts_must_match=config["transcripts_must_match"],
    )
    for model_config in config["models"]:
        model_name = model_config["name"]
        logger.info(f"Start alignment head selection for {model_name}  ")

        model_path = model_config["path"]
        model = configure_model(model_path, config["device"])
        logger.info(f"Loaded model {model_name} from {model_path}.")

        processor, _, _ = get_processor_config_and_genconfig(model_path)
        alignment_head_generator = get_alignment_head_generator(
            config["head_selection_num_heads"], config["head_selection_num_layers"]
        )

        for index_, alignment_heads in enumerate(
            tqdm(alignment_head_generator, desc=f"Processing {model_name}")
        ):
            logger.info(f"Testing alignment head config {index_}")
            configure_model_generation(model, model_name, processor, alignment_heads, 7)

            labels = load_labels(
                config["dataset_base_path"],
                config["head_selection_dataset"],
                "test",
                config["head_selection_num_samples"],
            )
            logger.info(
                f"Loaded {len(labels)} ground truth"
                f" labels from {config['dataset_base_path']}"
            )
            ground_truth = prepare_ground_truth(labels)
            asr_pipeline = setup_pipeline(model, processor, config["device"])

            audio_paths = [
                str(
                    config["dataset_base_path"]
                    / config["head_selection_dataset"]
                    / label["audio"]
                )
                for label in labels
            ]
            try:
                predictions, success_ids = process_audio_files(
                    audio_paths, asr_pipeline
                )
                ground_truth = [ground_truth[i] for i in success_ids]

                predicted = [
                    (
                        prediction["text"],
                        convert_timestamps_from_transformers_pipe_to_TimestampedOutput(
                            prediction["chunks"]
                        ),
                    )
                    for prediction in predictions
                ]

                new_predictions = adjust_pauses(predicted)
                seg_metrics, _ = batch_evaluate_segmentation(
                    references=ground_truth,
                    predictions=new_predictions,
                    eval_config=eval_config,
                )

                results.append(
                    {
                        "model": model_name,
                        "heads": alignment_heads,
                        f'F1_collar.{config["head_selection_collar"]}': seg_metrics.f1_score,
                        f'avg_iou_collar.{config["head_selection_collar"]}': seg_metrics.avg_iou,
                    }
                )
            except Exception as e:
                logger.info(e)

    return results


def perform_ablation_study(
    config: dict[str, Any], head_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    results = []
    head_df = pd.DataFrame(head_results)
    eval_config = EvaluationConfig(  # type: ignore
        collar=config["head_selection_collar"],
        transcripts_must_match=config["transcripts_must_match"],
    )
    for model_config in tqdm(config["models"]):
        logger.info(f"Working on {model_config}")
        model_name = model_config["name"]
        model_path = model_config["path"]
        model = configure_model(model_path, config["device"])
        processor, _, _ = get_processor_config_and_genconfig(model_path)

        for dataset in config["test_datasets"]:
            logger.info(f"Working on {dataset}")
            for add_noise in config["add_noise"]:
                for median_filter_width in config["median_filter_widths"]:
                    for num_heads in config["num_heads"]:
                        for legacy in [True, False]:
                            logger.info(
                                f"Current configuration: add_noise={add_noise}, "
                                f"median_filter_width={median_filter_width}, "
                                f"num_heads={num_heads}, legacy={legacy}"
                            )
                            top_heads_dict = top_heads_by_f1(
                                head_df,
                                num_heads,
                                config["head_selection_collar"],
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
                                config["dataset_base_path"],
                                dataset,
                                split="test",
                                limit=config["experiment_num_samples"],
                            )
                            model.generation_config.legacy = legacy
                            ground_truth = prepare_ground_truth(labels)
                            asr_pipeline = setup_pipeline(
                                model, processor, config["device"]
                            )
                            audio_paths = [
                                str(
                                    config["dataset_base_path"]
                                    / dataset
                                    / label["audio"]
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
                            ground_truth = [ground_truth[i] for i in success_ids]

                            predicted = [
                                (
                                    prediction["text"],
                                    convert_timestamps_from_transformers_pipe_to_TimestampedOutput(
                                        prediction["chunks"]
                                    ),
                                )
                                for prediction in predictions
                            ]
                            for threshold in config["pause_thresholds"]:
                                new_predictions = adjust_pauses(
                                    predicted, threshold=threshold
                                )
                                for collar in config["collars"]:
                                    (
                                        seg_metrics,
                                        seg_metrics_list,
                                    ) = (
                                        seg_metrics,
                                        _,
                                    ) = batch_evaluate_segmentation(
                                        references=ground_truth,
                                        predictions=new_predictions,
                                        eval_config=eval_config,
                                    )
                                    addendum = "_noise" if add_noise else ""
                                    model_name_addendum = (
                                        "legacy" if legacy else "regular"
                                    )
                                    seg_metrics_list_as_dict = [
                                        converter.unstructure(metrics)
                                        for metrics in seg_metrics_list
                                    ]
                                    results.append(
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
                                            "metrics": seg_metrics_list_as_dict,
                                        }
                                    )

    return results


def perform_baseline_study(config: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_output = []
    eval_config = EvaluationConfig(  # type: ignore
        collar=config["head_selection_collar"],
        transcripts_must_match=config["transcripts_must_match"],
    )
    device = config["device"]
    c_type = "float16" if "cuda" in device else "float32"
    for i, model_config in enumerate(
        [ModelConfig.TIMESTAMPED_WHISPER, ModelConfig.WHISPER_X]
    ):
        logger.info(f"Running baseline for {model_config}")
        model = (
            WhisperXModel(**model_config, device=device, compute_type=c_type)
            if i == 1
            else WhisperTimestamped(**model_config, device=device)
        )
        for dataset in config["test_datasets"]:
            logger.info(f"Running baseline for {dataset}")
            for add_noise in config["add_noise"]:
                labels = load_labels(
                    config["dataset_base_path"],
                    dataset,
                    split="test",
                    limit=config["experiment_num_samples"],
                )
                audio_paths = [
                    str(config["dataset_base_path"] / dataset / label["audio"])
                    for label in labels
                ]
                if add_noise:
                    audio_paths = [
                        add_gaussian_noise_to_audio(path, random.randint(1, 8))
                        for path in audio_paths
                    ]
                logger.info(f"Running baseline for {dataset}")
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
                    if config["dataset_base_path"] / dataset / label["audio"]
                    not in error_ids
                ]
                ground_truth = prepare_ground_truth(labels)
                for collar in config["collars"]:
                    seg_metrics, seg_metrics_list = batch_evaluate_segmentation(
                        ground_truth, timestamped_outputs, eval_config=eval_config
                    )
                    addendum = "_noise" if add_noise else ""
                    seg_metrics_list_as_dict = [
                        converter.unstructure(metrics) for metrics in seg_metrics_list
                    ]
                    ground_truth_as_dict = [
                        (trans_, ts_.to_json()) for trans_, ts_ in ground_truth
                    ]
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
                            "predictions": [output.to_json() for output in outputs],
                            "gts": ground_truth_as_dict,
                            "metrics": seg_metrics_list_as_dict,
                        }
                    )

    return baseline_output


def analyze_results(
    baseline_results: list[dict[str, Any]], output_path: Path, collar: float
) -> None:
    baseline_df = pd.DataFrame(baseline_results)
    filtered_df = baseline_df[baseline_df["Collar"] == collar]

    result_df = filtered_df.loc[
        filtered_df.groupby(["Model", "Dataset"])["F1 Score"].idxmax()
    ]

    pivoted_df = result_df.pivot(
        index="Dataset", columns="Model", values=["F1 Score", "Avg IOU"]
    )

    pivoted_df.columns = [f"{col[1]}_{col[0]}" for col in pivoted_df.columns]
    pivoted_df.reset_index(inplace=True)

    pivoted_df.to_csv(output_path / f"baseline_results_{collar}.csv", index=False)


def visualize_results(
    ablation_results: list[dict[str, Any]],
    output_path: Path,
    visualization_config: dict[str, Any],
) -> None:
    df = pd.DataFrame(ablation_results)
    # Todo: fix code here
    df["matching_indexes"] = df.apply(find_matching_indexes, axis=1)
    common_idx_df = (
        df.groupby(["Model", "Dataset"])["matching_indexes"]
        .agg(common_indexes)
        .reset_index()
    )
    df = pd.merge(df, common_idx_df, on=["Model", "Dataset"], suffixes=("", "_common"))
    df["F1 Score clean"] = df.apply(
        lambda row: add_score_metrics(row, "f1_score"), axis=1
    )
    df["Avg IOU clean"] = df.apply(
        lambda row: add_score_metrics(row, "avg_iou"), axis=1
    )

    plot_path = output_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    plot_f1_iou_vs_heads(df, visualization_config, plot_path)
    plot_average_f1_iou_vs_heads(df, visualization_config, plot_path)
    plot_f1_iou_vs_threshold(df, visualization_config, plot_path)
    plot_f1_vs_collar(df, visualization_config, plot_path)
    plot_f1_vs_median_filter_width(df, visualization_config, plot_path)


def plot_f1_iou_vs_heads(
    df: pd.DataFrame, config: dict[str, Any], plot_path: Path
) -> None:
    for dataset in config["visualization_datasets"]:
        collar = config["collar"]
        threshold = config["threshold"]
        median_filter_width = config["median_filter_width"]
        filtered_df = df[
            (df["Collar"] == collar)
            & (df["MedianFilterWidth"] == median_filter_width)
            & (df["Threshold"] == threshold)
            & (df["Model"].isin(config["models_to_compare"]))
            & (df["Dataset"] == dataset)
        ]

        sns.set(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 8))

        color_f1, color_iou = "tab:blue", "tab:green"
        marker_f1, marker_iou = "o", "X"

        sns.lineplot(
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
        ax1.legend(title="F1 Score", bbox_to_anchor=(1.12, 1), loc="upper left")

        ax2 = ax1.twinx()
        sns.lineplot(
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
        ax2.legend(title="Avg IOU", bbox_to_anchor=(1.12, 0.85), loc="upper left")

        fig.suptitle(
            f"Comparison of F1 Score and Avg IOU vs. Number of Heads\n"
            f"Collar={collar}, Dataset: {dataset}",
            fontsize=16,
        )
        plt.tight_layout()

        plt.savefig(
            plot_path
            / f"F1_IOU_vs_heads_collar_{collar}_dataset_{dataset}_threshold_{threshold}_medianfilterwidth_{median_filter_width}.png"
        )
        plt.close(fig)


def plot_average_f1_iou_vs_heads(
    df: pd.DataFrame, config: dict[str, Any], plot_path: Path
) -> None:
    collar = 0.1
    filtered_df = df[
        (df["Collar"] == collar)
        & (df["MedianFilterWidth"] == 3)
        & (df["Threshold"] == 0.1)
        & (df["Model"].isin(config["models_to_compare"]))
    ]
    grouped_df = (
        filtered_df.groupby(["Model", "num_heads"])
        .agg(
            {
                "F1 Score": "mean",
                "Avg IOU": "mean",
            }
        )
        .reset_index()
    )

    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = ["tab:blue", "tab:green"]

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

    ax1.legend(title="Model (F1 Score)", bbox_to_anchor=(1.12, 1), loc="upper left")
    ax2.legend(title="Model (Avg IOU)", bbox_to_anchor=(1.12, 0.85), loc="upper left")

    fig.suptitle(
        f"Average F1 Score and Avg IOU vs. Number of Heads Across Datasets\n"
        f"Collar: {collar}",
        fontsize=16,
    )
    plt.tight_layout()

    plt.savefig(plot_path / f"Average_F1_IOU_vs_heads_collar_{collar}.png")
    plt.close(fig)


def plot_f1_iou_vs_threshold(
    df: pd.DataFrame, config: dict[str, Any], plot_path: Path
) -> None:
    for dataset in config["visualization_datasets"]:
        collar = 0.1
        filtered_df = df[
            (df["Collar"] == collar)
            & (df["Dataset"] == dataset)
            & (df["MedianFilterWidth"] == 3)
            & (df["num_heads"] == 3)
            & (df["Model"].isin(["crisperWhisper++"]))
        ]
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

        sns.set(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 8))

        colors = ["tab:blue", "tab:green"]

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

        ax1.legend(title="Model (F1 Score)", bbox_to_anchor=(1.12, 1), loc="upper left")
        ax2.legend(
            title="Model (Avg IOU)", bbox_to_anchor=(1.12, 0.85), loc="upper left"
        )

        fig.suptitle(
            f"Average F1 Score and Avg IOU vs. Threshold\n"
            f"Dataset: {dataset}, Collar: {collar}",
            fontsize=16,
        )
        plt.tight_layout()

        plt.savefig(
            plot_path / f"F1_IOU_vs_threshold_dataset_{dataset}_collar_{collar}.png"
        )
        plt.close(fig)


def plot_f1_vs_collar(
    df: pd.DataFrame, config: dict[str, Any], plot_path: Path
) -> None:
    for dataset in config["visualization_datasets"]:
        filtered_df = df[
            (df["Threshold"] == 0.1)
            & (df["num_heads"] == 3)
            & (df["Dataset"] == dataset)
            & (df["MedianFilterWidth"] == 3)
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        for model, group in filtered_df.groupby("Model"):
            ax.plot(group["Collar"], group["F1 Score"], label=model, marker="o")

        ax.set_xlabel("Collar")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"F1 Score by Collar for Each Model\nDataset: {dataset}")
        ax.legend(title="Model")

        plt.savefig(plot_path / f"F1_vs_collar_dataset_{dataset}.png")
        plt.close(fig)


def plot_f1_vs_median_filter_width(
    df: pd.DataFrame, config: dict[str, Any], plot_path: Path
) -> None:
    for dataset in config["visualization_datasets"]:
        collar = 0.1
        filtered_df = df[
            (df["Threshold"] == 0.1)
            & (df["num_heads"] == 3)
            & (df["Dataset"] == dataset)
            & (df["Collar"] == collar)
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        for model, group in filtered_df.groupby("Model"):
            ax.plot(
                group["MedianFilterWidth"], group["F1 Score"], label=model, marker="o"
            )

        ax.set_xlabel("Median Filter Width")
        ax.set_ylabel("F1 Score")
        ax.set_title(
            f"F1 Score by Median Filter Width for Each Model\nDataset: {dataset}, Collar: {collar}"
        )
        ax.legend(title="Model")

        plt.savefig(
            plot_path
            / f"F1_vs_median_filter_width_dataset_{dataset}_collar_{collar}.png"
        )
        plt.close(fig)


def save_results(path: Path, data: Any) -> None:
    with path.open("wb" if path.suffix == ".pickle" else "w") as f:
        if path.suffix == ".json":
            json.dump(data, f, ensure_ascii=False, indent=2)
        elif path.suffix == ".pickle":
            pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def main() -> None:
    experiment_config = load_experiment_config()
    output_path = Path(experiment_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    head_results = process_alignment_heads(experiment_config)
    save_results(output_path / "head_results.json", head_results)

    ablation_results = perform_ablation_study(experiment_config, head_results)
    save_results(output_path / "legacy_vs_hf_ablations.json", ablation_results)

    baseline_results = perform_baseline_study(experiment_config)
    save_results(output_path / "baseline_ablations.json", baseline_results)

    for collar in experiment_config["collars"]:
        analyze_results(baseline_results, output_path, collar)
    # Todo: fix code in visualize_results before running below snippet
    # for collar in experiment_config["collars"]:
    #     for threshold in experiment_config["pause_thresholds"]:
    #         for median_filter_width in experiment_config["median_filter_widths"]:
    #             visualization_config = {
    #                 "models_to_compare": [
    #                     model_["name"] for model_ in experiment_config["models"]
    #                 ],
    #                 "visualization_datasets": experiment_config["test_datasets"],
    #                 "collar": collar,
    #                 "median_filter_width": median_filter_width,
    #                 "threshold": threshold,
    #             }
    #
    #             visualize_results(ablation_results, "plots", visualization_config)


if __name__ == "__main__":
    main()
