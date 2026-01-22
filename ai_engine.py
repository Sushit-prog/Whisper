import os
import numpy as np
import torch
import soundfile as sf
from transformers import pipeline
import ollama


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


print("Loading Whisper...")
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    framework="pt",
    device="cpu"
)

def process_audio_file(audio_path: str):
    # --- TRANSCRIBE ---
    audio_np, sample_rate = sf.read(audio_path, dtype='float32')
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    if sample_rate != 16000:
        from torchaudio.functional import resample
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        audio_tensor = resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
        audio_np = audio_tensor.squeeze().numpy()
        sample_rate = 16000

    result = transcriber(
        {"raw": audio_np, "sampling_rate": sample_rate},
        return_timestamps=True
    )
    transcript = " ".join(chunk["text"] for chunk in result["chunks"])

   
    print("Generating summary with Phi-3...")
    prompt = f"""
    You are a professional meeting assistant. Summarize clearly:
    - Key decisions
    - Action items (with owners if mentioned)
    Keep under 100 words.

    Transcript:
    {transcript}
    """
    summary_resp = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': prompt}])
    summary = summary_resp['message']['content']

    return transcript, summary
