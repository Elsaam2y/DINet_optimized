import glob
import math
import os
import time

import numpy as np
import torch

# from deep_speech import DeepSpeech
import torchaudio

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Using CPU")


class Wav2VecFeatureExtractor:
    def __init__(self):
        self.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        # Build the model and load the pretrained weights
        self.model = self.bundle.get_model().to(device)

    def compute_audio_feature(self, audio_path):
        # Load the audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)

        # Move the audio waveform to the GPU
        waveform = waveform.to(device)

        # Resample the audio to the expected sampling rate
        resampler = torchaudio.transforms.Resample(
            sample_rate, self.bundle.sample_rate
        ).to(device)
        waveform = resampler(waveform)

        # Infer the label probability distribution (emissions)
        emissions, _ = self.model(waveform)
        # ignore the second channel
        emissions = emissions[0, ::2]

        # Move the emissions back to the CPU if needed
        emissions = emissions.to("cpu")

        return emissions.detach().numpy()
