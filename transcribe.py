import argparse
import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def transcribe_audio(file_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"  # You can change this to a different model if needed

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(file_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("--f", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.f):
        print(f"Error: The file '{args.f}' does not exist.")
        sys.exit(1)

    try:
        transcription = transcribe_audio(args.f)
        print("Transcription:")
        print(transcription["text"])
    except Exception as e:
        print(f"An error occurred while transcribing the audio: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
