## Experiments 🔬

This section outlines the experiments conducted to evaluate the performance of cross-attention heads and various parameters relevant to Dynamic Time Warping (DTW).

### Head Selection
Initially, we conduct experiments where each cross-attention head is evaluated independently to determine its effectiveness in alignment when used alone. We will use the this evaluation in the following experiments.

### Ablation Studies
We perform a series of ablation studies to investigate several factors that may influence the performance of our alignment method:

- **Median Filter Width**: We explore the impact of varying the width of the median filter used to preprocess the matrix for DTW. The width of this filter is a crucial parameter that could significantly affect alignment accuracy.

- **Number of Alignment Heads**: The study includes examining how the number of heads selected for alignment affects performance. Heads are selected on a greedy basis.

- **Noise Robustness**: We assess the robustness of alignment heads against noise by introducing Gaussian noise with varying Signal to Noise Ratios (SNRs) into the audio samples.

- **Pause Heuristic**: To address the issue of unnaturally long durations attributed to the space token by DTW, we experiment with a heuristic that splits the allotted time for the space token between the preceding and following words. This experiment ablates different thresholds for time splitting to refine the handling of pauses in speech.

These experiments aim to optimize the parameters and strategies used in our DTW-based alignment process, ensuring more accurate and robust results.


## Run Experiments

To reuse our experimentation code or run your own experiments you should structure your datasets like this:

The `timed_datasets` directory contains multiple datasets, each with its own set of audio files and corresponding labels.json . Below is the structure of the datasets:

- `timed_datasets`: The root directory containing all timed datasets.
  - `timed_dataset_1`: A timed dataset directory.
    - `audio`: Directory containing all audio files for `timed_dataset_1`.
      - `example_audio_1.wav`: A example audio file belonging to the dataset.
    - `labels.json`: A JSON file containing labels for the audio files in `timed_dataset_1`.
  - `timed_dataset_2`: Another dataset directory.
    - `audio`: Directory containing audio files for `timed_dataset_2`.
      - `example_audio_2.wav`: An example audio file.
      - `example_audio_3.wav`: Another example audio file.
      - `example_audio_4.wav`: Another example audio file.
    - `labels.json`: A JSON file containing labels for the audio files in `timed_dataset_2`.


### Dataset structure

```plaintext
timed_datasets
├── timed_dataset_1
│   ├── audio
│   │   └── example_audio_1.wav
│   └── labels.json
└── timed_dataset_2
    ├── audio
    │   ├── example_audio_2.wav
    │   ├── example_audio_3.wav
    │   └── example_audio_4.wav
    └── labels.json

```

The structure for a labels.json file looks like this and includes timing information.

```plaintext
[
    {
        "audio": "audio/example_audio_1.wav",
        "split": "test",
        "duration_in_s": 2.9,
        "dataset_name": "timed_dataset_1",
        "transcript": "This is a example transcript",
        "labels": [
            {
                "word": "This",
                "starttime": 0.75,
                "endtime": 1.06
            },
            {
                "word": "is",
                "starttime": 1.2,
                "endtime": 1.55
            },
            {
                "word": "a",
                "starttime": 1.55,
                "endtime": 2.19
            },
            {
                "word": "example",
                "starttime": 2.19,
                "endtime": 2.41
            },
            {
                "word": "transcript",
                "starttime": 2.41,
                "endtime": 2.77
            },

        ]
    }
]
```
When your dataset(s) are stored in the correct format you should be able to easily run the experiments as defined in the Segmentation_experiments.ipynb. Make sure to adjust your Experiment configuration accordingly and run your own experiments.
