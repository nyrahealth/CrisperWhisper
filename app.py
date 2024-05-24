import streamlit as st
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio.transforms as T
from scipy.io import wavfile
import numpy as np
import io
import torchaudio
from streamlit_mic_recorder import mic_recorder
import moviepy.editor as mp
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Streamlit app for speech transcription.')
parser.add_argument('--model_id', type=str, required=True, help='Path to the model directory')
args = parser.parse_args()
model_id = args.model_id

# Set up device and data type for processing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor from the specified path
@st.cache_resource
def load_model_and_processor(model_id):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    model.generation_config.median_filter_width = 3
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

model, processor = load_model_and_processor(model_id)

# Setup the pipeline
@st.cache_resource
def setup_pipeline(_model, _processor):
    return pipeline(
        "automatic-speech-recognition",
        model=_model,
        tokenizer=_processor.tokenizer,
        feature_extractor=_processor.feature_extractor,
        max_new_tokens=800,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

pipe = setup_pipeline(model, processor)

def wav_to_black_mp4(wav_path, output_path, fps=25):
    waveform, sample_rate = torchaudio.load(wav_path)
    duration = waveform.shape[1] / sample_rate
    audio = mp.AudioFileClip(wav_path)
    black_clip = mp.ColorClip((256, 250), color=(0, 0, 0), duration=duration)
    final_clip = black_clip.set_audio(audio)
    final_clip.write_videofile(output_path, fps=fps)


def timestamps_to_srt(timestamps):
    vtt_content = "WEBVTT\n\n"
    for word in timestamps:
        start_time, end_time = word["timestamp"]
        start_time_str = f"{int(start_time // 3600)}:{int(start_time // 60 % 60):02d}:{start_time % 60:06.3f}".replace('.', '.')
        end_time_str = f"{int(end_time // 3600)}:{int(end_time // 60 % 60):02d}:{end_time % 60:06.3f}".replace('.', '.')
        vtt_content += f"{start_time_str} --> {end_time_str}\n{word['text']}\n\n"
    return vtt_content


def process_audio_bytes(audio_bytes):
    audio_stream = io.BytesIO(audio_bytes)
    sr, y = wavfile.read(audio_stream)
    y = y.astype(np.float32)
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std
    transform = T.Resample(sr, 16000)
    waveform = transform(torch.unsqueeze(torch.tensor(y_normalized), 0))
    waveform_to_save = transform(torch.unsqueeze(torch.tensor(y_normalized), 0))
    torchaudio.save('sample.wav', waveform_to_save, sample_rate=16000)
    return waveform

def transcribe(audio_bytes):
    waveform = process_audio_bytes(audio_bytes)
    transcription = pipe(waveform[0, :].numpy(), return_timestamps="word")
    return transcription

# Streamlit app interface
st.title("CrisperWhisper++ 🦻")
st.subheader("Caution when using. Make sure you can handle the crispness. ⚠️")
st.write("🎙️ Record an audio to transcribe or 📁 upload an audio file.")

# Audio recorder component
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    format="wav",
    callback=None,
    args=(),
    kwargs={},
    key=None
)

audio_bytes = audio['bytes'] if audio else None

# Audio file upload handling
audio_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    audio_bytes = audio_file.getvalue()

if audio_bytes:
    try:
        transcription = transcribe(audio_bytes)
        vtt = timestamps_to_srt(transcription['chunks'])
        
        with open("subtitles.vtt", "w") as file:
            file.write(vtt)
            
        wav_to_black_mp4('sample.wav', 'video.mp4')
        
        st.video('video.mp4', subtitles='subtitles.vtt')
        st.subheader("Transcription:")
        st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <p style="font-size: 16px; color: #333;">{transcription['text']}</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")

# Footer
st.markdown("""
    <hr>
    <footer>
        <p style="text-align: center;">© 2024 nyra health GmbH</p>
    </footer>
    """, unsafe_allow_html=True)
