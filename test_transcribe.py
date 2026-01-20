import os
import numpy as np
import torch
import soundfile as sf
from transformers import pipeline


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Loading Whisper model...")
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    framework="pt",
    device="cpu"
)

audio_path = "test_meeting.wav"


audio_np, sample_rate = sf.read(audio_path, dtype='float32')
if audio_np.ndim > 1:
    audio_np = np.mean(audio_np, axis=1)


if sample_rate != 16000:
    from torchaudio.functional import resample
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
    audio_tensor = resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
    audio_np = audio_tensor.squeeze().numpy()
    sample_rate = 16000


print("Transcribing (may take a minute)...")
result = transcriber(
    {"raw": audio_np, "sampling_rate": sample_rate},
    return_timestamps=True  # ← Required for >30 sec
)


full_text = " ".join(chunk["text"] for chunk in result["chunks"])
print("\n🎯 TRANSCRIPT:")
print(full_text)