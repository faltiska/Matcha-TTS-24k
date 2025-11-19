"""Vocos vocoder wrapper for Matcha-TTS."""

import torch
import torch.nn as nn
from vocos import Vocos

class VocosWrapper(nn.Module):
    """
    Wrapper for Vocos vocoder to make it compatible with Matcha-TTS interface.
    Vocos is a mel-spectrogram vocoder that can be used as a drop-in replacement for HiFiGAN.
    """

    def __init__(self, vocos_model):
        super().__init__()
        self.vocos = vocos_model

    def forward(self, mel):
        """
        Forward pass through Vocos vocoder.
        
        Args:
            mel: Mel-spectrogram tensor of shape (batch, n_mels, time)
        
        Returns:
            audio: Audio waveform tensor of shape (batch, 1, samples)
        """
        # Vocos expects mel-spectrogram input
        # The model processes mel spectrograms and outputs audio
        with torch.no_grad():
            audio = self.vocos.decode(mel)
        
        # Ensure output shape matches HiFiGAN format (batch, 1, samples)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        return audio

    def eval(self):
        """Set model to evaluation mode."""
        self.vocos.eval()
        return self

    def to(self, device):
        """Move model to specified device."""
        self.vocos = self.vocos.to(device)
        return super().to(device)


def load_vocoder(device):
    """
    Load the pretrained Vocos model from https://huggingface.co/BSC-LT/vocos-mel-22khz
    
    Args:
        device: Device to load the model on (cpu/cuda)
    
    Returns:
        VocosWrapper instance
    """
    
    vocos = Vocos.from_pretrained("BSC-LT/vocos-mel-22khz")
    vocos = vocos.to(device)
    vocos.eval()
    
    wrapper = VocosWrapper(vocos)
    wrapper.eval()
    
    return wrapper
