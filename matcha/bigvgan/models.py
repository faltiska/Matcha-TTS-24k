"""Vocos vocoder wrapper for Matcha-TTS."""

import torch
import torch.nn as nn
import bigvgan

class BigVGANWrapper(nn.Module):
    """
    Wrapper for Vocos vocoder to make it compatible with Matcha-TTS interface.
    Vocos is a mel-spectrogram vocoder that can be used as a drop-in replacement for HiFiGAN.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        """
        Forward pass through the vocoder.
        
        Args:
            mel: Mel-spectrogram tensor of shape (batch, n_mels, time)
        
        Returns:
            audio: Audio waveform tensor of shape (batch, 1, samples)
        """
        with torch.inference_mode():
            audio = self.model(mel)
        
        return audio

def load_vocoder(device):
    """
    Load the pretrained BigVGAN model from https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_fmax8k_256x
    
    Args:
        device: Device to load the model on (cpu/cuda)
    
    Returns:
        BigVGANWrapper instance
    """

    model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_fmax8k_256x', use_cuda_kernel=True)
    model.remove_weight_norm()
    model = model.eval().to(device)
    
    return BigVGANWrapper(model)