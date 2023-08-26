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


# This process requires GPU
class Wav2VecFeatureExtractor:
    def __init__(self):
        self.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        # Build the model and load the pretrained weights
        self.model = self.bundle.get_model().to(device)

    # fix OOM issues by processing long audio files into smaller chunks
    def compute_audio_feature(self, audio_path, chunk_duration=10):
        # Load the audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)

        # Calculate the number of samples in each chunk
        chunk_samples = int(chunk_duration * sample_rate)

        # Initialize a list to store the results for each chunk
        all_emissions = []

        # Iterate over the audio waveform in chunks
        for i in range(0, waveform.size(1), chunk_samples):
            chunk = waveform[:, i : i + chunk_samples]

            # Move the chunk to the same device as the model
            chunk = chunk.to(device)

            # Resample the chunk
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.bundle.sample_rate
            ).to(device)
            chunk = resampler(chunk)

            # Infer the label probability distribution (emissions) for the chunk
            emissions, _ = self.model(chunk)
            emissions = emissions[0, ::2]

            # Move the emissions back to the CPU
            emissions = emissions.to("cpu")

            all_emissions.append(emissions.detach().numpy())

        return np.concatenate(all_emissions, axis=0)
